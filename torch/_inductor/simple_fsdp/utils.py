import time
from collections import defaultdict
from enum import IntEnum
from typing import Dict, List, Set, Tuple, Union

import torch
import torch.distributed as dist
import torch.utils._pytree as pytree
from torch.utils._mode_utils import no_dispatch

from .. import ir, scheduler
from ..config import simplefsdp
from ..utils import get_gpu_dram_gbps, is_collective, is_wait


class NodeType(IntEnum):
    ALL_GATHER = 0
    COMPUTE = 1
    REDUCE_SCATTER = 2
    AG_WAIT = 3
    RS_WAIT = 4


def compute_node_users(
    snodes: List["scheduler.BaseSchedulerNode"],
) -> Tuple[
    Dict["scheduler.BaseSchedulerNode", Set["scheduler.BaseSchedulerNode"]],
    Dict["scheduler.BaseSchedulerNode", Set["scheduler.BaseSchedulerNode"]],
]:
    """
    Compute the inverse users and users of each node
    """
    buf_to_snode: Dict[str, scheduler.BaseSchedulerNode] = {}
    for node in snodes:
        if isinstance(node, scheduler.FusedSchedulerNode):
            for x in node.snodes:
                for buf in x.get_outputs():
                    buf_to_snode[buf.get_name()] = node

        for buf in node.get_outputs():
            buf_to_snode[buf.get_name()] = node

    inverse_users = {}
    keys = list(buf_to_snode.keys())
    for node in snodes:
        dep_list = []
        for dep in node.unmet_dependencies:
            if dep.name in keys:
                dep_list.append(buf_to_snode[dep.name])
        inverse_users.update({node: set(dep_list)})

    node_users: Dict[scheduler.BaseSchedulerNode, Set[scheduler.BaseSchedulerNode]] = (
        defaultdict(set)
    )
    for node, node_inverse_users in inverse_users.items():
        for inverse_user in node_inverse_users:
            node_users[inverse_user].add(node)

    return inverse_users, node_users


def _check_ir_node_fsdp(ir_node: "ir.Operation") -> bool:
    """
    Determine if the AG/RS node is for FSDP or TP
    """
    if not simplefsdp.tp_enabled:
        return True

    is_fsdp = False
    ir_node_origins = list(getattr(ir_node, "origins", None))

    if len(ir_node_origins) == 0:
        # bucketed AG and RS doesn't have origins, but they are created by FSDP
        is_fsdp = True

    for n in ir_node_origins:
        meta_data = n.meta.get("stack_trace", {})
        # TODO(ruisizhang123): hack to get FSDP node (the FSDP AG/RS are created from torch_spmd)
        if "parametrization" in meta_data:
            is_fsdp = True
    return is_fsdp


def _get_ir_node_type(ir_node: "ir.Operation") -> NodeType:
    """
    Determine the type of a ir node
    """
    if isinstance(ir_node, ir._WaitKernel):
        # Determine if the wait node is waiting for ALL_GATHER or REDUCE_SCATTER
        ir_op_overload = getattr(ir_node.inputs[0], "op_overload", None)
        if (
            ir_op_overload == torch.ops._c10d_functional.all_gather_into_tensor.default
            and _check_ir_node_fsdp(ir_node.inputs[0])
        ):
            return NodeType.AG_WAIT
        elif (
            ir_op_overload == torch.ops._c10d_functional.reduce_scatter_tensor.default
            and _check_ir_node_fsdp(ir_node.inputs[0])
        ):
            return NodeType.RS_WAIT
    elif isinstance(ir_node, ir._CollectiveKernel):
        # Determine if the collective kernel is for ALL_GATHER or REDUCE_SCATTER
        ir_op_overload = getattr(ir_node, "op_overload", None)
        if (
            ir_op_overload == torch.ops._c10d_functional.all_gather_into_tensor.default
            and _check_ir_node_fsdp(ir_node)
        ):
            return NodeType.ALL_GATHER
        elif (
            ir_op_overload == torch.ops._c10d_functional.reduce_scatter_tensor.default
            and _check_ir_node_fsdp(ir_node)
        ):
            return NodeType.REDUCE_SCATTER

    return NodeType.COMPUTE


def get_node_type(node: "scheduler.BaseSchedulerNode") -> NodeType:
    """
    Determine the NodeType of a node
    """
    if isinstance(node, scheduler.FusedSchedulerNode):
        # Only compute nodes are fused
        return NodeType.COMPUTE

    if isinstance(node, scheduler.GroupedSchedulerNode):
        # [Only for bucketing]: newly created AG and RS are grouped as GroupedSchedulerNode
        child_nodes_type = [
            _get_ir_node_type(n) for n in [node.snodes[0].node, node.snodes[-1].node]
        ]
        if child_nodes_type[0] in [NodeType.AG_WAIT, NodeType.RS_WAIT]:
            return child_nodes_type[0]
        elif child_nodes_type[1] in [NodeType.ALL_GATHER, NodeType.REDUCE_SCATTER]:
            return child_nodes_type[1]
        else:
            return NodeType.COMPUTE

    return _get_ir_node_type(node.node)


def _get_benchmark_runtime(node) -> List[float]:
    """
    Returns estimated op runtime in nanoseconds (ns)
    """
    # Communication kernel benchmark
    if is_collective(node.node):
        # the AG/RS's communication time will be dynamically re-estimated in greedy bucketing algorithm
        # we skip the estimation here to improve efficiency
        return [0, 0]
    elif is_wait(node.node):
        # wait is not profiled in GPU
        return [0, 0]

    # Compute kernel benchmark
    if isinstance(node.node, ir.ComputedBuffer):
        return [node.get_read_write_buffers_sizes() / get_gpu_dram_gbps() * 1e-6, 0]
    elif isinstance(node.node, ir.ExternKernel):
        cls = node.node.__class__
        func = scheduler.kernel_name_to_op.get(
            getattr(node.node, "python_kernel_name", ""), None
        )
        if func is None:
            func = scheduler.kernel_name_to_op.get(
                str(getattr(node.node, "op_overload", "")), None
            )
        if func is not None:
            # Don't use in_kernel_invocation_manager(fake_mode) as we want to do
            # REAL compute (not with meta device)
            inp_impls = {}

            fake_inputs = [
                ir.ir_node_to_tensor(input, guard_shape=False)
                for input in node.node.inputs
            ]
            flat_args, args_spec = pytree.tree_flatten((fake_inputs, node.node.kwargs))
            new_kwargs = node.node.fill_non_provided_args(fake_inputs, node.node.kwargs)

            with no_dispatch():

                def to_real_tensor(e):
                    if not isinstance(e, torch.Tensor):
                        return e
                    if torch.is_floating_point(e):
                        out = torch.rand_like(e, device=e.device)
                    else:
                        out = torch.ones_like(e, device=e.device)
                    if e.is_sparse:
                        out._coalesced_(e.is_coalesced())
                    inp_impls[id(out)] = e
                    return out

                flat_args = [to_real_tensor(a) for a in flat_args]
                args, kwargs = pytree.tree_unflatten(flat_args, args_spec)
                r = func(*args, **kwargs)
                num_iters = 3
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                cpu_start = time.time()
                total_memory = 0
                start_event.record(torch.cuda.current_stream())
                for _ in range(num_iters):
                    r = None
                    r = func(*args, **kwargs)
                    memory = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
                    total_memory += memory
                end_event.record(torch.cuda.current_stream())
                cpu_end = time.time()
                torch.cuda.synchronize()
                cpu_time = (cpu_end - cpu_start) / 1000
                total_op_time = start_event.elapsed_time(end_event) - cpu_time
                mean_op_time = total_op_time / num_iters
                mean_memory = total_memory / num_iters
                del flat_args
            return [mean_op_time, mean_memory]
        else:
            return [0, 0]
    else:
        return [0, 0]


def _get_runtime_dict(
    snode: List["scheduler.BaseSchedulerNode"],
) -> Dict[str, List[Union[str, float, float]]]:
    total_node = 0
    run_time_dict = {}
    for n in snode:
        if isinstance(n, scheduler.FusedSchedulerNode):
            total_time, total_memory = 0, 0
            buffer_op_name = [str(i) for i in n.snodes[0].node.origins]
            buffer_op_name = "".join(buffer_op_name)
            for subnode in n.snodes:
                stime, smemory = _get_benchmark_runtime(subnode)
                total_time += stime
                total_memory += smemory
            run_time_dict[n.get_name()] = [buffer_op_name, total_time, total_memory]
        elif isinstance(n, scheduler.GroupedSchedulerNode):
            total_time, total_memory = 0, 0
            buffer_op_name = [str(i) for i in n.snodes[0].node.origins]
            buffer_op_name = "".join(buffer_op_name)
            for subnode in n.snodes:
                stime, smemory = _get_benchmark_runtime(subnode)
                total_time += stime
                total_memory += smemory
            run_time_dict[n.get_name()] = [buffer_op_name, total_time, total_memory]
        else:
            if isinstance(n.node, ir._WaitKernel):
                run_time_dict[n.get_name()] = ["", 0, 0]
            else:
                python_kernel = getattr(n.node, "python_kernel_name", None)
                if python_kernel is not None:
                    buffer_op_name = n.node.python_kernel_name
                else:
                    buffer_op_name = [str(i) for i in n.node.origins]
                    buffer_op_name = "".join(buffer_op_name)

                benchmark_op = _get_benchmark_runtime(n)
                run_time_dict[n.get_name()] = [buffer_op_name] + benchmark_op
    return run_time_dict


def profile_nodes(
    snode: List["scheduler.BaseSchedulerNode"],
) -> Dict[str, List[Union[str, float, float]]]:
    current_rank = dist.get_rank()
    objects = [None]
    if not simplefsdp.pp_enabled:
        if current_rank == 0:
            run_time_dict = _get_runtime_dict(snode)
            objects = [run_time_dict]
        dist.broadcast_object_list(objects, src=0)
    else:
        # broadcast runtime dict from rank 0 (on each PP stage) to make sure every rank receives the same runtime estimation results.
        broadcast_lists = simplefsdp.device_mesh
        receive_runtime_dict = {}
        rank_group_dict = {}
        send_runtime_rank = []

        for broadcast_list in broadcast_lists:
            if not simplefsdp.tp_enabled:
                source_rank = broadcast_list[0]
                flatten_list = broadcast_list
            else:
                source_rank = broadcast_list[0][0]
                flatten_list = [i for sublist in broadcast_list for i in sublist]

            rank_group_dict[source_rank] = dist.new_group(ranks=flatten_list)
            send_runtime_rank.append(source_rank)
            for subnode in flatten_list:
                receive_runtime_dict[subnode] = source_rank

        if current_rank in send_runtime_rank:
            run_time_dict = _get_runtime_dict(snode)
            objects = [run_time_dict]
        dist.broadcast_object_list(
            objects,
            src=receive_runtime_dict[current_rank],
            group=rank_group_dict[receive_runtime_dict[current_rank]],
        )
    assert objects[0] is not None
    return objects[0]
