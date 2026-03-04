import math

# Calibrated coefficients (β parameters from system identification)
beta_bw = 1.002877
beta_dynamic = 2.081349
beta_storage = 13.495403

hdfs_overhead = 1.047


def predict_stage_duration(
    T_base: float,
    N_tasks: int,
    N_partitions: int,
    input_read_gb: float,
    window_max_bytes: int,
    mtu_bytes: int,
    spark_num_executors: int,
    cores_per_executor: int,
    network_latency_ms: float,
    B_link: float,
):

    if input_read_gb <= 0.0:
        return T_base, {}

    RTT = network_latency_ms / 1000.0

    E = spark_num_executors * cores_per_executor
    N_active = min(N_tasks, E)

    N_waves = max(1.0, N_tasks / E)

    rho = N_partitions / N_tasks

    # TCP window parameters
    W_init = 10 * (mtu_bytes * 8) / 1e9
    W_max = window_max_bytes * 8 / 1e9 / 2

    # Baseline demand rate
    b_base = hdfs_overhead * input_read_gb * 8 / T_base

    # Flow throughput
    B_flow = W_max / RTT

    # Effective bandwidth
    B_eff = min(B_link, N_active * B_flow)

    # ---- Bandwidth component ----

    T_bw_base = T_base * max(1.0, b_base / B_eff)
    T_bw = beta_bw * T_bw_base

    # ---- Storage latency component ----

    T_storage_base = rho * N_waves * RTT
    T_storage = beta_storage * T_storage_base

    # ---- Slow start component ----

    rate_per_flow = min(b_base, B_eff) / N_active
    W_target = rate_per_flow * RTT

    nu = max(0.0, RTT * math.log2(W_target / W_init)) if W_target > 0 else 0.0

    T_dynamic_base = rho * N_waves * nu
    T_dynamic = beta_dynamic * T_dynamic_base

    # ---- Total predicted hybrid stage time ----

    T_hat_hybrid = T_bw + T_storage + T_dynamic

    return T_hat_hybrid, {
        "T_bw": T_bw,
        "T_storage": T_storage,
        "T_dynamic": T_dynamic,
        "T_bw_base": T_bw_base,
        "T_storage_base": T_storage_base,
        "T_dynamic_base": T_dynamic_base,
    }