from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from .. import scheduler
from ..comm_analysis import estimate_bucketed_nccl_collective_runtime, get_predicted_node_comm
from ..config import simplefsdp
from .bucket import merge_ag_wait, merge_allgather, merge_reducescatter, merge_rs_wait
from .utils import compute_node_users, get_node_type, NodeType


@dataclass
class CollectiveInfo:
    ag_inv_dep: List["scheduler.BaseSchedulerNode"] = field(default_factory=list)
    ag_wait: List["scheduler.BaseSchedulerNode"] = field(default_factory=list)
    compute: List["scheduler.BaseSchedulerNode"] = field(default_factory=list)
    compute_time: float = 0
    compute_memory: float = 0

    # additional configs for backward
    reduce_scatter: List["scheduler.BaseSchedulerNode"] = field(default_factory=list)
    rs_wait: List["scheduler.BaseSchedulerNode"] = field(default_factory=list)
    rs_wait_dep: List["scheduler.BaseSchedulerNode"] = field(default_factory=list)


def greedy_check(
    current_comp,
    current_mem,
    current_ag,
    node_mem,
    memory_constraint,
    is_first_ag,
    last_rs=0,
):
    """
    Check if the current node satisfy the greedy bucketing rule
        Return False: the current node satisfy the greedy rule and should not start a new bucket
        Return True: the current node doesn't satisfy the greedy rule and should start a new bucket
    Return True if:
        1. the current node is the first AG in the graph
        2. compute memory constraint is not satisfied
        3. the current node's communication time cannot be overlapped with last bucket's compute time
    """
    if is_first_ag:
        return False

    if current_mem + node_mem > memory_constraint:
        return True

    if current_ag + last_rs > current_comp:
        return True

    return False


def get_collective_info(
    snodes: List["scheduler.BaseSchedulerNode"],
    run_time_dict: Dict[str, List[Union[str, float, float]]],
    is_backward: Optional[bool] = False,
) -> Dict["scheduler.BaseSchedulerNode", CollectiveInfo]:
    """
    Get the information of all_gather and reduce_scatter
    """
    inverse_users, node_users = compute_node_users(snodes)
    front_nodes = []
    all_gather = None
    collective_info_dict = {}

    # pick up the computes that don't have corresponding all_gather
    for node in snodes:
        users_type = [get_node_type(i) for i in list(node_users[node])]
        if NodeType.ALL_GATHER in users_type or get_node_type(node) == NodeType.ALL_GATHER:
            break
        front_nodes.append(node)

    for node in snodes[len(front_nodes) :]:
        if node.get_name() in run_time_dict:
            _, run_time, memory = run_time_dict[node.get_name()]
        else:
            run_time, memory = 0, 0

        if get_node_type(node) == NodeType.ALL_GATHER:
            # A CollectiveInfo consists of 8 parts:
            # [Node]: (1) ag_inv_dep: The node ALL_GATHER depends on for read-in; (2) ag_wait: ALL_GATHER's wait nodes; (3) compute: compute nodes ALL_GATHER fetches.
            # （4） reduce_scatter: REDUCE_SCATTE that reads from compute nodes; (5) rs_wait: reduce_scatter's wait nodes; (6) rs_wait_dep: rs_wait's dep nodes.
            # [Estimation Num.]: (1) compute_time: The estimated compute time; (2) compute_memory: The estimated meory for computation.
            collective_info = CollectiveInfo(
                ag_wait=list(node_users[node]),
            )
            collective_info_dict[node] = collective_info
            all_gather = node

            if len(front_nodes) > 0:
                collective_info_dict[all_gather].ag_inv_dep.extend(front_nodes)
                front_nodes = []
            collective_info_dict[all_gather].ag_inv_dep.extend(
                list(inverse_users[node])
            )

        if get_node_type(node) == NodeType.COMPUTE:
            # if the compute node is the inverse user of AG and user of RS_Wait, we should group them with the next AG or last RS_Wait
            users_type = [get_node_type(i) for i in list(node_users[node])]
            if NodeType.ALL_GATHER in users_type:
                continue
            if is_backward:
                inverse_users_type = [
                    get_node_type(i) for i in list(inverse_users[node])
                ]
                if NodeType.RS_WAIT in inverse_users_type:
                    continue
            collective_info_dict[all_gather].compute.append(node)
            collective_info_dict[all_gather].compute_time += run_time
            collective_info_dict[all_gather].compute_memory += memory

        if is_backward:
            if get_node_type(node) == NodeType.REDUCE_SCATTER:
                # rs_dep --> rs --> rs_wait --> rs_wait_dep
                collective_info_dict[all_gather].reduce_scatter.append(node)

            if get_node_type(node) == NodeType.RS_WAIT:
                collective_info_dict[all_gather].rs_wait.append(node)
                collective_info_dict[all_gather].rs_wait_dep.extend(
                    list(node_users[node])
                )

    # make sure all nodes are indexed in collective_info_dict
    count = 0
    for key, value in collective_info_dict.items():
        count += (
            len(value.ag_inv_dep)
            + len(value.ag_wait)
            + len(value.compute)
            + len(value.reduce_scatter)
            + len(value.rs_wait)
            + len(value.rs_wait_dep)
        )
        count += 1
    assert count == len(snodes)

    return collective_info_dict


def get_greedy_bucket_plan(
    sched: "scheduler.Scheduler",
    snodes: List["scheduler.BaseSchedulerNode"],
    collective_info_dict: Dict["scheduler.BaseSchedulerNode", CollectiveInfo],
    is_backward: Optional[bool] = False,
    use_analytical_model: Optional[bool] = False,
) -> List["scheduler.BaseSchedulerNode"]:
    """
    Greedy Bucket ALL_GATHER and reduce_scatter
    """
    if use_analytical_model:
        communication_estimation = estimate_bucketed_nccl_collective_runtime
    else:
        communication_estimation = get_predicted_node_comm

    result_list = []
    all_gather_list = []
    ag_inv_dep_list = []
    ag_wait_list = []
    compute_list = []

    current_comp = 0  # compute time in step_i
    current_mem = 0  # memory in step_i
    last_rs = 0  # reduce scatter time in step_i, by default, it is 0 in forward
    next_comp = 0  # compute time in step_(i+1), derived from all gather in step_i
    is_first_ag = True
    memory_constraint = simplefsdp.memory_constraint

    if is_backward:
        reduce_scatter_list = []
        rs_wait_list = []
        rs_wait_dep_list = []
        current_rs = 0  # reduce scatter time in step_(i+1), derived from all gather in step_i

    for idx, node in enumerate(snodes):
        if get_node_type(node) == NodeType.ALL_GATHER:
            # TODO(ruisizhang123): the memory_constraint is defined manually, we need to figure out a better way to get the memory_constraint
            # the first all gather will not pass greedy_check, because is_first_ag is False
            if greedy_check(
                current_comp,
                current_mem,
                communication_estimation(all_gather_list + [node]),
                collective_info_dict[node].compute_memory,
                memory_constraint,
                is_first_ag,
                last_rs,
            ):
                # merge all_gather
                assert len(all_gather_list) > 0
                print("communication_estimation(all_gather_list + [node])", communication_estimation(all_gather_list))
                merged_all_gather, ag_buffer = merge_allgather(sched, all_gather_list, ag_inv_dep_list)
                merged_ag_wait = merge_ag_wait(
                    sched, ag_wait_list, all_gather_list, ag_buffer
                )
                for n in ag_inv_dep_list + [merged_all_gather, merged_ag_wait]:
                    if n not in result_list:
                        result_list.append(n)
                compute_list = [i for i in compute_list if i not in result_list]
                result_list.extend(compute_list)

                # merge reduce_scatter
                if is_backward and len(reduce_scatter_list) > 0:
                    (merged_reduce_scatter, rs_buffer, copy_in_size) = (
                        merge_reducescatter(sched, reduce_scatter_list)
                    )
                    merged_rs_wait = merge_rs_wait(
                        sched,
                        rs_wait_list,
                        reduce_scatter_list,
                        rs_buffer,
                        copy_in_size,
                    )

                    for n in [merged_reduce_scatter, merged_rs_wait] + rs_wait_dep_list:
                        if n not in result_list:
                            result_list.append(n)

                # clear the list for bucketing
                all_gather_list = [node]
                ag_inv_dep_list = collective_info_dict[node].ag_inv_dep
                ag_wait_list = collective_info_dict[node].ag_wait
                compute_list = collective_info_dict[node].compute

                # clear the number for greedy
                current_comp = next_comp
                current_mem = collective_info_dict[node].compute_memory
                next_comp = collective_info_dict[node].compute_time

                if is_backward:
                    reduce_scatter_list = collective_info_dict[node].reduce_scatter
                    rs_wait_list = collective_info_dict[node].rs_wait
                    rs_wait_dep_list = collective_info_dict[node].rs_wait_dep
                    last_rs = current_rs
                    current_rs = communication_estimation(
                        reduce_scatter_list, is_ag=False
                    )
            else:
                # update the list for bucketing
                all_gather_list.append(node)
                ag_inv_dep_list.extend(collective_info_dict[node].ag_inv_dep)
                ag_wait_list.extend(collective_info_dict[node].ag_wait)
                compute_list.extend(collective_info_dict[node].compute)

                # update the number for greedy
                current_mem += collective_info_dict[node].compute_memory
                next_comp += collective_info_dict[node].compute_time

                # the first AG is not bucketed
                is_first_ag = False

                if is_backward:
                    reduce_scatter_list.extend(
                        collective_info_dict[node].reduce_scatter
                    )
                    rs_wait_list.extend(collective_info_dict[node].rs_wait)
                    rs_wait_dep_list.extend(collective_info_dict[node].rs_wait_dep)
                    current_rs = communication_estimation(
                        reduce_scatter_list, is_ag=False
                    )

    if len(all_gather_list) > 0:
        merged_all_gather, ag_buffer = merge_allgather(sched, all_gather_list)
        merged_ag_wait = merge_ag_wait(sched, ag_wait_list, all_gather_list, ag_buffer)
        for n in ag_inv_dep_list + [merged_all_gather, merged_ag_wait]:
            if n not in result_list:
                result_list.append(n)
    compute_list = [i for i in compute_list if i not in result_list]
    result_list.extend(compute_list)

    # merge reduce_scatter
    if is_backward and len(reduce_scatter_list) > 0:
        (merged_reduce_scatter, rs_buffer, copy_in_size) = merge_reducescatter(
            sched, reduce_scatter_list
        )
        merged_rs_wait = merge_rs_wait(
            sched, rs_wait_list, reduce_scatter_list, rs_buffer, copy_in_size
        )

        for n in [merged_reduce_scatter, merged_rs_wait] + rs_wait_dep_list:
            if n not in result_list:
                result_list.append(n)
    return result_list


def bucket_by_greedy(
    sched: "scheduler.Scheduler",
    snodes: List["scheduler.BaseSchedulerNode"],
    run_time_dict: Dict[str, List[Union[str, float]]],
    is_backward: Optional[bool] = False,
    use_analytical_model: Optional[bool] = False,
) -> List["scheduler.BaseSchedulerNode"]:
    """
    Greedy bucket ALL_GATHER and ag_wait
    """
    collective_info_dict = get_collective_info(snodes, run_time_dict, is_backward)
    result_list = get_greedy_bucket_plan(
        sched, snodes, collective_info_dict, is_backward, use_analytical_model
    )
    return result_list
