import os
import pickle
import functools
import math
from enum import IntEnum
from typing import List, Any, Callable
import numpy as np

import sympy

import torch

from . import ir
from .config import simplefsdp
from .utils import get_dtype_size, sympy_product
from .virtualized import V


class NCCL_COLL(IntEnum):
    ALL_REDUCE = 0
    ALL_GATHER = 1
    REDUCE_SCATTER = 2


class NVIDIA_GPU_TYPE(IntEnum):
    VOLTA = 0
    AMPERE = 1
    HOPPER = 2


@functools.lru_cache
def get_gpu_type() -> NVIDIA_GPU_TYPE:
    gpu_info = torch.utils.collect_env.get_gpu_info(torch.utils.collect_env.run) or ""
    if "V100" in gpu_info:
        return NVIDIA_GPU_TYPE.VOLTA
    elif "A100" in gpu_info:
        return NVIDIA_GPU_TYPE.AMPERE
    elif "H100" in gpu_info:
        return NVIDIA_GPU_TYPE.HOPPER
    else:
        # for other gpu types, assume Ampere
        return NVIDIA_GPU_TYPE.AMPERE


def get_collective_type(node: ir.IRNode) -> NCCL_COLL:
    if not isinstance(node, ir._CollectiveKernel):
        raise ValueError(f"node is not a collective kernel: {node}")

    kernel_name = node.python_kernel_name
    assert kernel_name is not None
    if "all_reduce" in kernel_name:
        return NCCL_COLL.ALL_REDUCE
    elif "all_gather" in kernel_name:
        return NCCL_COLL.ALL_GATHER
    elif "reduce_scatter" in kernel_name:
        return NCCL_COLL.REDUCE_SCATTER
    else:
        raise ValueError(f"Unsupported collective kernel: {kernel_name}")


def get_collective_input_size_bytes(node: ir.IRNode) -> int:
    sz_bytes = 0
    for inp in node.inputs:  # type: ignore[attr-defined]
        numel = sympy_product(inp.layout.size)
        if isinstance(numel, sympy.Integer):
            # For ease of testing
            numel = int(numel)
        else:
            numel = V.graph.sizevars.size_hint(numel, fallback=0)
        sz_bytes += numel * get_dtype_size(inp.layout.dtype)
    return sz_bytes


def get_collective_group_size(node: ir.IRNode) -> int:
    if type(node) == ir._CollectiveKernel:
        from torch.distributed.distributed_c10d import _get_group_size_by_name

        return _get_group_size_by_name(node.constant_args[-1])
    else:
        raise TypeError(f"Unsupported collective type: {node}")


####################################################################################################################
# The following code and constants are adapted from https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc #
####################################################################################################################


class NCCL_HW(IntEnum):
    NVLINK = 0
    PCI = 1
    NET = 2


class NCCL_ALGO(IntEnum):
    TREE = 0
    RING = 1


class NCCL_PROTO(IntEnum):
    # The ordering and enum values here matches original in
    # https://github.com/NVIDIA/nccl/blob/0b083e52096c387bad7a5c5c65b26a9dca54de8c/src/include/devcomm.h#L28
    # For difference between these protocols, see https://github.com/NVIDIA/nccl/issues/281#issuecomment-571816990
    LL = 0  # Low-latency
    # LL128 = 1   # Low-latency 128-byte
    # SIMPLE = 2


# Latencies in us
# len(NCCL_ALGO) x len(NCCL_PROTO)
# NOTE: use array instead of tensor to prevent incompatibility with fake mode
baseLat = [
    # Tree
    [
        6.8,  # LL
    ],
    # Ring
    [
        6.6,  # LL
    ],
]

# Latencies in us
# len(NCCL_HW) x len(NCCL_ALGO) x len(NCCL_PROTO)
hwLat = [
    # NVLINK
    [
        [0.6],  # Tree (LL)
        [0.6],  # Ring (LL)
    ],
    # PCI
    [
        [1.0],  # Tree (LL)
        [1.0],  # Ring (LL)
    ],
    # NET
    [
        [5.0],  # Tree (LL)
        [2.7],  # Ring (LL)
    ],
]


# LL128 max BW per channel
llMaxBws = [
    # Volta-N1/Intel-N2/Intel-N4
    [
        39.0,
        39.0,
        20.4,
    ],
    # Ampere-N1/AMD-N2/AMD-N4
    [
        87.7,
        22.5,  # avg of ring & tree
        19.0,
    ],
    # Hopper-N1/AMD-N2/AMD-N4
    [
        87.7,
        22.5,  # avg of ring & tree
        19.0,
    ],
]


def estimate_nccl_collective_runtime(node: ir.IRNode) -> float:
    """
    Returns estimated NCCL collective runtime in nanoseconds (ns).

    The following heuristics are copied from https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc.
    We aim to estimate the runtime as accurately as possible.

    Assumptions:
    - only ring algorithm (NCCL_ALGO_RING) is used
    - only Low-Latency protocol (NCCL_PROTO_LL) is used, i.e. Simple or LL128 is not used
    - 8 gpus per node  # TODO: Need to find a way to get accurate "gpus per node" and "# nodes" info.
    - collective is one of: allreduce, reducescatter, allgather
    """
    tensor_storage_size_bytes = get_collective_input_size_bytes(node)
    # Convert bytes to GB
    tensor_storage_size_GB = tensor_storage_size_bytes / 1024 / 1024 / 1024

    # Currently assumes each node has 8 gpus. And when >1 node is used, assumes each node uses all 8 gpus.
    # TODO: Need to find a way to get accurate "gpus per node" and "# nodes" info.
    num_gpus_per_node = 8
    group_size = get_collective_group_size(node)
    nNodes = math.ceil(group_size / num_gpus_per_node)
    nRanks = group_size  # this is total # of gpus globally that participate in this collective op

    if nRanks <= 1:
        return 0

    # Assumes ring algorithm
    nccl_algo = NCCL_ALGO.RING
    nccl_proto = NCCL_PROTO.LL
    coll = get_collective_type(node)

    # =============== bandwidth computation ===============
    # First compute bandwidth in GB/s; then at the end, convert it to GB/ns

    bwIntra = torch._inductor.config.intra_node_bw
    bwInter = torch._inductor.config.inter_node_bw

    compCapIndex = get_gpu_type()
    index2 = nNodes - 1 if nNodes <= 2 else 2
    # LL: for single node, we look at GPU type; for multi-node, we look at CPU type
    index1 = compCapIndex if nNodes == 1 else 0
    llMaxBw = llMaxBws[index1][index2]

    # NOTE: each step of ring algorithm is synchronized,
    # and is bottlenecked by the slowest link which is the inter-node interconnect.
    # hence when nNodes >= 2, bw is inter-node bandwidth.
    # NOTE: the original code in https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc
    # have this as `if nNodes <= 2` which seems wrong. Corrected it here.
    bw = bwIntra if nNodes == 1 else bwInter
    nChannels = 2  # Assume # channels is 2
    busBw = nChannels * bw

    # Various model refinements
    busBw = min(
        llMaxBw,
        busBw
        * (1.0 / 4.0 if (nNodes > 1 or coll == NCCL_COLL.ALL_REDUCE) else 1.0 / 3.0),
    )

    if coll == NCCL_COLL.ALL_REDUCE:
        nsteps = 2 * (nRanks - 1)
    elif coll in (NCCL_COLL.REDUCE_SCATTER, NCCL_COLL.ALL_GATHER):
        nsteps = nRanks - 1

    # Convert bus BW to algorithm BW (tensor bytes / algoBW = actual execution time)
    ratio = (1.0 * nRanks) / nsteps  # type: ignore[possibly-undefined]
    bandwidth = busBw * ratio
    # Convert GB/s to GB/ns
    bandwidth_GB_per_ns = bandwidth / 1e9

    # =============== latency computation ===============
    intraHw = NCCL_HW.NVLINK

    if coll == NCCL_COLL.ALL_REDUCE:
        if nNodes > 1:
            nInterSteps = 2 * nNodes
        else:
            nInterSteps = 0
    elif coll in (NCCL_COLL.REDUCE_SCATTER, NCCL_COLL.ALL_GATHER):
        nInterSteps = nNodes - 1

    # First compute latency in us; then at the end, convert it to ns
    latency = baseLat[nccl_algo][nccl_proto]
    intraLat = hwLat[intraHw][nccl_algo][nccl_proto]
    interLat = hwLat[NCCL_HW.NET][nccl_algo][nccl_proto]

    # Inter-node rings still have to launch nsteps * net overhead.
    netOverhead = 0.0
    if nNodes > 1:
        netOverhead = 1.0  # getNetOverhead(comm);
    intraLat = max(intraLat, netOverhead)
    latency += (nsteps - nInterSteps) * intraLat + nInterSteps * interLat  # type: ignore[possibly-undefined]
    # Convert us to ns
    latency_ns = latency * 1e3

    # =============== final result ===============
    transport_ns = tensor_storage_size_GB / bandwidth_GB_per_ns
    return transport_ns + latency_ns


################################################################################################################
# The above code and constants are adapted from https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc #
################################################################################################################


def estimate_bucketed_nccl_collective_runtime(
    nodes: List["scheduler.BaseSchedulerNode"], is_ag=True
) -> float:
    if len(nodes) == 0:
        return 0
    
    # Function to estimate the runtime of bucketed AG/RS
    num_gpus_per_node = 8
    group_size = get_collective_group_size(nodes[0].node)
    nNodes = math.ceil(group_size / num_gpus_per_node)
    nRanks = group_size  # this is total # of gpus globally that participate in this collective op

    if nRanks <= 1:
        return 0

    # Assumes ring algorithm
    nccl_algo = NCCL_ALGO.RING
    nccl_proto = NCCL_PROTO.LL
    coll = get_collective_type(nodes[0].node)

    # =============== bandwidth computation ===============
    # First compute bandwidth in GB/s; then at the end, convert it to GB/ns

    bwIntra = torch._inductor.config.intra_node_bw
    bwInter = torch._inductor.config.inter_node_bw

    compCapIndex = get_gpu_type()
    index2 = nNodes - 1 if nNodes <= 2 else 2
    # LL: for single node, we look at GPU type; for multi-node, we look at CPU type
    index1 = compCapIndex if nNodes == 1 else 0
    llMaxBw = llMaxBws[index1][index2]

    # NOTE: each step of ring algorithm is synchronized,
    # and is bottlenecked by the slowest link which is the inter-node interconnect.
    # hence when nNodes >= 2, bw is inter-node bandwidth.
    # NOTE: the original code in https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc
    # have this as `if nNodes <= 2` which seems wrong. Corrected it here.
    bw = bwIntra if nNodes == 1 else bwInter
    nChannels = 2  # Assume # channels is 2
    busBw = nChannels * bw

    # Various model refinements
    busBw = min(
        llMaxBw,
        busBw
        * (1.0 / 4.0 if (nNodes > 1 or coll == NCCL_COLL.ALL_REDUCE) else 1.0 / 3.0),
    )

    if coll == NCCL_COLL.ALL_REDUCE:
        nsteps = 2 * (nRanks - 1)
    elif coll in (NCCL_COLL.REDUCE_SCATTER, NCCL_COLL.ALL_GATHER):
        nsteps = nRanks - 1

    # Convert bus BW to algorithm BW (tensor bytes / algoBW = actual execution time)
    ratio = (1.0 * nRanks) / nsteps  # type: ignore[possibly-undefined]
    bandwidth = busBw * ratio
    # Convert GB/s to GB/ns
    bandwidth_GB_per_ns = bandwidth / 1e9

    # =============== latency computation ===============
    intraHw = NCCL_HW.NVLINK

    if coll == NCCL_COLL.ALL_REDUCE:
        if nNodes > 1:
            nInterSteps = 2 * nNodes
        else:
            nInterSteps = 0
    elif coll in (NCCL_COLL.REDUCE_SCATTER, NCCL_COLL.ALL_GATHER):
        nInterSteps = nNodes - 1

    # First compute latency in us; then at the end, convert it to ns
    latency = baseLat[nccl_algo][nccl_proto]
    intraLat = hwLat[intraHw][nccl_algo][nccl_proto]
    interLat = hwLat[NCCL_HW.NET][nccl_algo][nccl_proto]

    # Inter-node rings still have to launch nsteps * net overhead.
    netOverhead = 0.0
    if nNodes > 1:
        netOverhead = 1.0  # getNetOverhead(comm);
    intraLat = max(intraLat, netOverhead)
    latency += (nsteps - nInterSteps) * intraLat + nInterSteps * interLat  # type: ignore[possibly-undefined]
    # Convert us to ns
    latency_ns = latency * 1e3

    total_tensor_storage_size_bytes = 0
    for node in nodes:
        tensor_storage_size_bytes = get_collective_input_size_bytes(node.node)
        # Convert bytes to GB
        tensor_storage_size_GB = tensor_storage_size_bytes / 1024 / 1024 / 1024
        total_tensor_storage_size_bytes += tensor_storage_size_GB

    # =============== final result ===============
    transport_ns = total_tensor_storage_size_bytes / bandwidth_GB_per_ns
    # adjust the estimated communication time by a coefficient
    if is_ag:
        return (transport_ns + latency_ns) * 1e-6 * simplefsdp.ag_comm_time_multiplier
    return (transport_ns + latency_ns) * 1e-6 * simplefsdp.rs_comm_time_multiplier


aten = torch.ops.aten
c10d = torch.ops.c10d
_c10d_functional = torch.ops._c10d_functional
_c10d_functional_autograd = torch.ops._c10d_functional_autograd
_dtensor = torch.ops._dtensor


def listdir(p: str):
    return [os.path.join(p, f) for f in os.listdir(p)]



# substitute for your param path and num_gpus per node constants
# gpus_per_node = 4
# path = "/n/netscratch/idreos_lab/Everyone/FASRC_COLLECTIVE_MODELS"
gpus_per_node = 4
path = "/home/ruisi/model_params"

modeled_collectives = ["all_reduce", "all_gather", "reduce_scatter"]


# read saved parameters
def read_bw_params(path: str):
    interbw_params = np.load(os.path.join(path, "interbw_params.npy"))
    intrabw_params = np.load(os.path.join(path, "intrabw_params.npy"))
    return interbw_params, intrabw_params


def read_latency_params(path: str) -> tuple[float, float]:
    latency_params = np.load(os.path.join(path, "latency.npy"))
    return latency_params


def read_collective_params(path: str, collective: str, two_d: bool):
    with open(
        os.path.join(path, f'{collective}_{"2D" if two_d else "1D"}_params.pkl'), "rb"
    ) as f:
        return pickle.load(f)


# read constants and predict
interbw_params, intrabw_params = read_bw_params(path)
internode_latency, intranode_latency = read_latency_params(path)
saved_params: dict[tuple[str, bool], Any] = {
    ("all_reduce", False): read_collective_params(path, "all_reduce", False),
    ("all_reduce", True): read_collective_params(path, "all_reduce", True),
    ("all_gather", False): read_collective_params(path, "all_gather", False),
    ("all_gather", True): read_collective_params(path, "all_gather", True),
    ("reduce_scatter", False): read_collective_params(path, "reduce_scatter", False),
    ("reduce_scatter", True): read_collective_params(path, "reduce_scatter", True),
}


# bandwidth models
def sigmoid(x, L, x0, k) -> float:
    y = L / (1 + np.exp(-k * (x - x0)))
    return y


def log_sigmoid(x, L, x0, k) -> float:
    return sigmoid(np.log(x), L, x0, k)


def inter_bw(x) -> float:
    return log_sigmoid(x, *interbw_params)


def intra_bw(x) -> float:
    return log_sigmoid(x, *intrabw_params)


# collective models
def all_reduce_model(
    data_size: float, N: int, internode_only: bool = False
) -> tuple[float, float]:
    if internode_only:
        N //= gpus_per_node
        time_tree = (
            (2 * data_size) / (inter_bw(data_size + 1) / gpus_per_node) if N > 1 else 0
        )
        latency = (gpus_per_node - 1) * internode_latency
        return time_tree + latency, latency
    else:
        time_tree = (2 * data_size) / (inter_bw(data_size + 1)) if N > 1 else 0
        time_chain = (2 * data_size) / (intra_bw(data_size + 1))

        latency = (np.floor(np.log2(N)) + 1) * intranode_latency + (
            gpus_per_node - 1
        ) * internode_latency
        return time_tree + time_chain + latency, latency


def reduce_scatter_model(
    data_size: float, N: int, internode_only: bool = False
) -> tuple[float, float]:
    global internode_latency, intranode_latency
    n_nodes = N // gpus_per_node

    time_intra = ((N - n_nodes - 1) * data_size) / (
        (N - n_nodes) * intra_bw(data_size + 1)
    )
    time_inter = ((n_nodes - 1) * data_size) / ((n_nodes) * inter_bw(data_size + 1))

    latency = (
        n_nodes * internode_latency + (gpus_per_node - 1) * intranode_latency
        if not internode_only
        else n_nodes * internode_latency
    )

    if internode_only:
        return time_inter + latency, latency
    else:
        return time_intra + time_inter + latency, latency


def all_gather_model(
    data_size: float, N: int, internode_only: bool = False
) -> tuple[float, float]:
    global internode_latency, intranode_latency
    n_nodes = N // gpus_per_node

    time_intra = ((N - n_nodes - 1) * data_size) / (
        (N - n_nodes) * intra_bw(data_size + 1)
    )
    time_inter = ((n_nodes - 1) * data_size) / ((n_nodes) * inter_bw(data_size + 1))

    latency = (
        n_nodes * internode_latency + (gpus_per_node - 1) * intranode_latency
        if not internode_only
        else n_nodes * internode_latency
    )

    if internode_only:
        return time_inter + latency, latency
    else:
        return time_intra + time_inter + latency, latency


def broadcast_model(
    data_size: float, N: int, internode: bool = True
) -> tuple[float, float]:
    n_nodes = N // gpus_per_node

    if internode:
        return data_size / (inter_bw(data_size + 1) / gpus_per_node) + (
            n_nodes * internode_latency
        ), (n_nodes * internode_latency)
    else:
        return (
            data_size / intra_bw(data_size + 1)
            + ((gpus_per_node - 1) * intranode_latency)
        ), (gpus_per_node - 1) * intranode_latency


def scatter_model(
    data_size: float, N: int, internode: bool = True
) -> tuple[float, float]:
    n_nodes = N // gpus_per_node

    if internode:
        return data_size / (inter_bw(data_size + 1) / gpus_per_node) + (
            n_nodes * internode_latency
        ), (n_nodes * internode_latency)
    else:
        return (
            data_size / intra_bw(data_size + 1)
            + ((gpus_per_node - 1) * intranode_latency),
            (gpus_per_node - 1) * intranode_latency,
        )


def gather_model(
    data_size: float, N: int, internode: bool = True
) -> tuple[float, float]:
    n_nodes = N // gpus_per_node

    if internode:
        return data_size / (inter_bw(data_size + 1) / gpus_per_node) + (
            n_nodes * internode_latency
        ), (n_nodes * internode_latency)
    else:
        return (
            data_size / intra_bw(data_size + 1)
            + ((gpus_per_node - 1) * intranode_latency),
            (gpus_per_node - 1) * intranode_latency,
        )


def reduce_model(
    data_size: float, N: int, internode: bool = True
) -> tuple[float, float]:
    n_nodes = N // gpus_per_node

    if internode:
        return data_size / (inter_bw(data_size + 1) / gpus_per_node) + (
            n_nodes * internode_latency
        ), (n_nodes * internode_latency)
    else:
        return (
            data_size / intra_bw(data_size + 1)
            + ((gpus_per_node - 1) * intranode_latency),
            (gpus_per_node - 1) * intranode_latency,
        )


model_map: dict[str, Callable] = {
    "broadcast": broadcast_model,
    "all_reduce": all_reduce_model,
    "reduce": reduce_model,
    "all_gather": all_gather_model,
    "reduce_scatter": reduce_scatter_model,
    "gather": gather_model,
    "scatter": scatter_model,
    "send_recv": broadcast_model,  # TODO: Verify if this makes sense
}


def get_collective_model(func: torch._ops.OpOverload) -> Callable:
    broadcast_ops = {
        c10d.broadcast_.default,
        _c10d_functional.broadcast.default,
        _c10d_functional.broadcast_.default,
    }

    all_reduce_ops = {
        c10d.allreduce_.default,
        _c10d_functional.all_reduce.default,
        _c10d_functional.all_reduce_.default,
        _c10d_functional.all_reduce_coalesced.default,
        _c10d_functional.all_reduce_coalesced_.default,
    }

    reduce_ops = {c10d.reduce_.default}

    all_gather_ops = {
        c10d.allgather_.default,
        c10d._allgather_base_.default,
        c10d.alltoall_.default,
        c10d.alltoall_base_.default,
        _c10d_functional.all_to_all_single.default,
        _c10d_functional.all_gather_into_tensor.default,
        _c10d_functional_autograd.all_to_all_single.default,
        _dtensor.shard_dim_alltoall.default,
        _c10d_functional_autograd.all_gather_into_tensor.default,
        _c10d_functional.all_gather_into_tensor_coalesced.default,
        _c10d_functional.all_gather_into_tensor_out.default,
    }

    reduce_scatter_ops = {
        c10d.reduce_scatter_.default,
        c10d._reduce_scatter_base_.default,
        _c10d_functional.reduce_scatter_tensor.default,
        _c10d_functional.reduce_scatter_tensor_coalesced.default,
        _c10d_functional_autograd.reduce_scatter_tensor.default,
    }

    gather_ops = {c10d.gather_.default}
    scatter_ops = {c10d.scatter_.default}
    send_recv_ops = {c10d.send.default, c10d.recv_.default, c10d.recv_any_source_.default}

    # Map the function to the appropriate model based on the defined sets
    if func in broadcast_ops:
        return model_map["broadcast"]
    elif func in all_reduce_ops:
        return model_map["all_reduce"]
    elif func in reduce_ops:
        return model_map["reduce"]
    elif func in all_gather_ops:
        return model_map["all_gather"]
    elif func in reduce_scatter_ops:
        return model_map["reduce_scatter"]
    elif func in gather_ops:
        return model_map["gather"]
    elif func in scatter_ops:
        return model_map["scatter"]
    elif func in send_recv_ops:
        return model_map["send_recv"]

    raise ValueError(f"Unknown collective operation: {func}")


def predict_communication(
    collective: Any, data_size: float, N: int, internode_only: bool = False, analytical_mode: bool = False
) -> float:
    global saved_params
    if isinstance(collective, str):
        model_func = model_map[collective]
    else:
        model_func = get_collective_model(collective)

    if collective in modeled_collectives and not analytical_mode:
        analytical, latency = model_func(data_size, N, internode_only)
        min_params, straggle_params = saved_params[(collective, internode_only)]

        min_model = (
            min_params["Intercept"]
            + min_params["model"] * analytical
            + min_params["size"] * data_size
            + min_params["model:size"] * analytical * data_size
            + min_params["N"] * N
            + min_params["model:N"] * analytical * N
            + min_params["latency"] * latency
            + min_params["latency:size"] * data_size * latency
        )

        min_model = min_model if min_model > 0 else analytical

        straggle_model = (
            straggle_params["Intercept"]
            + straggle_params["np.log(size)"] * np.log(data_size)
            + straggle_params["N"] * N
        )

        model_pred = min_model * straggle_model
        return model_pred
    else:
        analytical, _ = model_func(data_size, N, internode_only)
        return analytical

def get_predicted_node_comm(
    nodes: List["scheduler.BaseSchedulerNode"], is_ag=True
) -> float:
    if len(nodes) == 0:
        return 0

    total_tensor_storage_size_bytes = 0
    for node in nodes:
        tensor_storage_size_bytes = get_collective_input_size_bytes(node.node)
        # Convert bytes to MB
        tensor_storage_size_MB = tensor_storage_size_bytes / 1024 / 1024
        total_tensor_storage_size_bytes += tensor_storage_size_MB

    group_size = get_collective_group_size(nodes[0].node)

    if is_ag:
        prediction = predict_communication("all_gather", total_tensor_storage_size_bytes, group_size, False)
    else:
        prediction = predict_communication("reduce_scatter", total_tensor_storage_size_bytes, group_size, False)

    return prediction
    
if __name__ == "__main__":
    prediction = predict_communication("all_gather", 711308800 / 2**20, 128, False)
    print(f"Predicted all_gather time: {prediction}")

    prediction = predict_communication("all_reduce", 422617600 / 2**20, 64, False)
    print(f"Predicted all_reduce time: {prediction}")

    prediction = predict_communication("reduce_scatter", 511308800 / 2**20, 256, False)
    print(f"Predicted reduce_scatter time: {prediction}")

