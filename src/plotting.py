import math
from pathlib import Path

import matplotlib.pyplot as plt

from utils import load_config, ensure_dir, read_csv_dicts, write_csv, minmax_norm


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def build_summary(plot_cfg):
    rows = []

    # -------- Dynamic: Early Exit --------
    early_exit = read_csv_dicts(plot_cfg["inputs"]["dynamic_early_exit_csv"])
    for r in early_exit:
        rows.append({
            "module": "dynamic",
            "variant": f"early_exit_thr_{r.get('threshold', '')}",
            "metric_top1": _safe_float(r.get("top1", 0.0)),
            "metric_top5": _safe_float(r.get("top5", 0.0)),
            "latency_ms": _safe_float(r.get("avg_latency_ms", 0.0)),
            "throughput_fps": _safe_float(r.get("throughput_fps", 0.0)),
            "avg_frames": _safe_float(r.get("avg_frames", 0.0)),
            "privacy_per": "",
            "privacy_flr": "",
            "entropy": _safe_float(r.get("avg_conf", 0.0)),  # early-exit CSV has avg_conf; keep in entropy column as proxy
            "comm_mb_round": "",
            "round": "",
            "note": "early_exit",
        })

    # -------- Dynamic: Frame Gating --------
    gating = read_csv_dicts(plot_cfg["inputs"]["dynamic_frame_gating_csv"])
    for r in gating:
        rows.append({
            "module": "dynamic",
            "variant": f"frame_gating_k_{r.get('k', '')}",
            "metric_top1": _safe_float(r.get("top1", 0.0)),
            "metric_top5": _safe_float(r.get("top5", 0.0)),
            "latency_ms": _safe_float(r.get("avg_latency_ms", 0.0)),
            "throughput_fps": "",  # not in this CSV
            "avg_frames": _safe_float(r.get("k", 0.0)),
            "privacy_per": "",
            "privacy_flr": "",
            "entropy": "",
            "comm_mb_round": "",
            "round": "",
            "note": "frame_gating",
        })

    # -------- Privacy: Visual PER --------
    vis = read_csv_dicts(plot_cfg["inputs"]["privacy_visual_csv"])
    # your visual_privacy_summary.csv should contain one row
    for r in vis:
        rows.append({
            "module": "privacy_visual",
            "variant": "visual_privacy",
            "metric_top1": "",
            "metric_top5": "",
            "latency_ms": "",
            "throughput_fps": "",
            "avg_frames": "",
            "privacy_per": _safe_float(r.get("per", "")) if r.get("per", "") != "" else "",
            "privacy_flr": "",
            "entropy": "",
            "comm_mb_round": "",
            "round": "",
            "note": "PER computed via face detector before/after",
        })

    # -------- Privacy: Feature (sigma / mask / FLR / entropy) --------
    feat = read_csv_dicts(plot_cfg["inputs"]["privacy_feature_csv"])
    # expects columns: sigma,mask_ratio,top1,top5,entropy,flr OR sigma,top1,top5,entropy
    for r in feat:
        sigma = r.get("sigma", "")
        mask = r.get("mask_ratio", r.get("mask", ""))
        variant = f"feature_sigma_{sigma}_mask_{mask}".replace("..", ".")
        rows.append({
            "module": "privacy_feature",
            "variant": variant,
            "metric_top1": _safe_float(r.get("top1", r.get("acc", 0.0))),
            "metric_top5": _safe_float(r.get("top5", 0.0)),
            "latency_ms": "",
            "throughput_fps": "",
            "avg_frames": "",
            "privacy_per": "",
            "privacy_flr": _safe_float(r.get("flr", "")) if r.get("flr", "") != "" else "",
            "entropy": _safe_float(r.get("entropy", 0.0)),
            "comm_mb_round": "",
            "round": "",
            "note": "feature_noise/masking + optional attacker FLR",
        })

    # -------- Federated: Accuracy vs comm --------
    fed = read_csv_dicts(plot_cfg["inputs"]["federated_csv"])
    for r in fed:
        rows.append({
            "module": "federated",
            "variant": f"fed_round_{r.get('round', '')}",
            "metric_top1": _safe_float(r.get("val_top1", 0.0)),
            "metric_top5": _safe_float(r.get("val_top5", 0.0)),
            "latency_ms": "",
            "throughput_fps": "",
            "avg_frames": "",
            "privacy_per": "",
            "privacy_flr": "",
            "entropy": "",
            "comm_mb_round": _safe_float(r.get("comm_mb_round", 0.0)),
            "round": _safe_float(r.get("round", 0.0)),
            "note": "FedAvg global validation per round",
        })

    return rows


def export_summary(plot_cfg, rows):
    out_path = plot_cfg["outputs"]["summary_csv"]
    header = [
        "module", "variant",
        "metric_top1", "metric_top5",
        "latency_ms", "throughput_fps", "avg_frames",
        "privacy_per", "privacy_flr", "entropy",
        "comm_mb_round", "round",
        "note"
    ]
    write_csv(out_path, header, rows)
    print(f"[INFO] Wrote summary CSV: {out_path}")


def fig17_dynamic_tradeoff(fig_dir, rows, dpi=200):
    # Early-exit: Top1 vs latency
    xs, ys, labels = [], [], []
    for r in rows:
        if r["module"] == "dynamic" and r["note"] == "early_exit":
            lat = _safe_float(r["latency_ms"], 0.0)
            acc = _safe_float(r["metric_top1"], 0.0)
            if lat > 0:
                xs.append(lat)
                ys.append(acc)
                labels.append(r["variant"])

    if len(xs) == 0:
        print("[INFO] Skip Fig17 (no early-exit data)")
        return

    plt.figure()
    plt.scatter(xs, ys)
    plt.xlabel("Average latency per batch (ms)")
    plt.ylabel("Top-1 accuracy")
    plt.title("Dynamic Inference Trade-off (Early Exit)")
    out = Path(fig_dir) / "Fig17_dynamic_early_exit_tradeoff.png"
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved figure: {out}")


def fig18_feature_privacy_tradeoff(fig_dir, rows, dpi=200):
    # Feature: sigma vs Top1, optionally group by mask
    data = []
    for r in rows:
        if r["module"] == "privacy_feature":
            v = r["variant"]
            # parse sigma/mask from variant
            # variant: feature_sigma_{sigma}_mask_{mask}
            try:
                parts = v.split("_")
                sigma = float(parts[2])
                mask = float(parts[4])
            except Exception:
                continue
            data.append((sigma, mask, _safe_float(r["metric_top1"], 0.0), _safe_float(r["privacy_flr"], float("nan"))))

    if len(data) == 0:
        print("[INFO] Skip Fig18 (no feature privacy data)")
        return

    # plot: sigma on x, top1 on y, multiple lines by mask
    masks = sorted(set([m for _, m, _, _ in data]))
    plt.figure()
    for m in masks:
        pts = sorted([(s, a) for s, mm, a, _ in data if abs(mm - m) < 1e-9], key=lambda x: x[0])
        if len(pts) == 0:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", label=f"mask={m:.2f}")
    plt.xlabel("Gaussian noise sigma")
    plt.ylabel("Top-1 accuracy")
    plt.title("Feature-level Privacy Trade-off (Accuracy vs Noise/Mask)")
    plt.legend()
    out = Path(fig_dir) / "Fig18_feature_privacy_tradeoff.png"
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved figure: {out}")

    # optional: FLR vs sigma (if available)
    flr_avail = [x for x in data if not math.isnan(x[3])]
    if len(flr_avail) > 0:
        plt.figure()
        for m in masks:
            pts = sorted([(s, flr) for s, mm, _, flr in flr_avail if abs(mm - m) < 1e-9], key=lambda x: x[0])
            if len(pts) == 0:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            plt.plot(xs, ys, marker="o", label=f"mask={m:.2f}")
        plt.xlabel("Gaussian noise sigma")
        plt.ylabel("FLR (attacker Top-1)")
        plt.title("Feature Leakage Rate under Feature Protection")
        plt.legend()
        out2 = Path(fig_dir) / "Fig18b_feature_flr_tradeoff.png"
        plt.savefig(out2, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Saved figure: {out2}")


def fig19_federated_tradeoff(fig_dir, rows, dpi=200):
    # Fed: round vs top1, and comm_mb_round vs top1
    rounds, accs, comms = [], [], []
    for r in rows:
        if r["module"] == "federated":
            rd = _safe_float(r["round"], 0.0)
            acc = _safe_float(r["metric_top1"], 0.0)
            comm = _safe_float(r["comm_mb_round"], 0.0)
            if rd > 0:
                rounds.append(rd)
                accs.append(acc)
                comms.append(comm)

    if len(rounds) == 0:
        print("[INFO] Skip Fig19 (no federated data)")
        return

    plt.figure()
    plt.plot(rounds, accs, marker="o")
    plt.xlabel("Communication rounds")
    plt.ylabel("Global Top-1 accuracy")
    plt.title("Federated Learning: Accuracy vs Rounds")
    out = Path(fig_dir) / "Fig19_federated_acc_vs_rounds.png"
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved figure: {out}")

    plt.figure()
    plt.scatter(comms, accs)
    plt.xlabel("Communication cost per round (MB)")
    plt.ylabel("Global Top-1 accuracy")
    plt.title("Federated Learning: Accuracy vs Communication Cost")
    out2 = Path(fig_dir) / "Fig19b_federated_acc_vs_comm.png"
    plt.savefig(out2, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved figure: {out2}")


def fig20_joint_pareto(fig_dir, rows, plot_cfg, dpi=200):
    """
    Joint trade-off:
    - choose representative points from:
      * dynamic early_exit (latency)
      * privacy feature (flr or per)
      * federated (comm)
    Build normalized cost and scatter cost vs accuracy.
    """
    w_lat = float(plot_cfg["joint_score"]["weights"]["w_latency"])
    w_comm = float(plot_cfg["joint_score"]["weights"]["w_comm"])
    w_priv = float(plot_cfg["joint_score"]["weights"]["w_privacy"])

    privacy_source = plot_cfg["joint_score"]["privacy_source"]

    # collect candidate points
    pts = []

    # dynamic points (use early_exit)
    for r in rows:
        if r["module"] == "dynamic" and r["note"] == "early_exit":
            acc = _safe_float(r["metric_top1"], 0.0)
            lat = _safe_float(r["latency_ms"], 0.0)
            if lat > 0:
                pts.append({
                    "name": f"dyn:{r['variant']}",
                    "acc": acc,
                    "lat": lat,
                    "comm": None,
                    "priv": None,
                })

    # privacy points (use feature)
    for r in rows:
        if r["module"] == "privacy_feature":
            acc = _safe_float(r["metric_top1"], 0.0)
            flr = r["privacy_flr"]
            priv = _safe_float(flr, 0.0) if flr != "" else None
            pts.append({
                "name": f"priv:{r['variant']}",
                "acc": acc,
                "lat": None,
                "comm": None,
                "priv": priv,
            })

    # federated points
    for r in rows:
        if r["module"] == "federated":
            acc = _safe_float(r["metric_top1"], 0.0)
            comm = _safe_float(r["comm_mb_round"], 0.0)
            if comm > 0:
                pts.append({
                    "name": f"fed:{r['variant']}",
                    "acc": acc,
                    "lat": None,
                    "comm": comm,
                    "priv": None,
                })

    if len(pts) == 0:
        print("[INFO] Skip Fig20 (no data)")
        return

    # Normalize each cost dimension among available values
    lat_vals = [p["lat"] for p in pts if p["lat"] is not None]
    comm_vals = [p["comm"] for p in pts if p["comm"] is not None]
    priv_vals = [p["priv"] for p in pts if p["priv"] is not None]

    lat_norm_map = {}
    if len(lat_vals) > 0:
        lat_norm = minmax_norm(lat_vals)
        for v, n in zip(lat_vals, lat_norm):
            lat_norm_map[v] = n

    comm_norm_map = {}
    if len(comm_vals) > 0:
        comm_norm = minmax_norm(comm_vals)
        for v, n in zip(comm_vals, comm_norm):
            comm_norm_map[v] = n

    priv_norm_map = {}
    if len(priv_vals) > 0:
        priv_norm = minmax_norm(priv_vals)
        for v, n in zip(priv_vals, priv_norm):
            priv_norm_map[v] = n

    # Build joint cost score
    xs_cost = []
    ys_acc = []
    for p in pts:
        lat_c = lat_norm_map.get(p["lat"], 0.0) if p["lat"] is not None else 0.0
        comm_c = comm_norm_map.get(p["comm"], 0.0) if p["comm"] is not None else 0.0
        priv_c = priv_norm_map.get(p["priv"], 0.0) if p["priv"] is not None else 0.0
        cost = w_lat * lat_c + w_comm * comm_c + w_priv * priv_c
        xs_cost.append(cost)
        ys_acc.append(p["acc"])

    plt.figure()
    plt.scatter(xs_cost, ys_acc)
    plt.xlabel("Normalized joint cost (lower is better)")
    plt.ylabel("Top-1 accuracy (higher is better)")
    plt.title("Joint Trade-off (Dynamic / Privacy / Federated)")
    out = Path(fig_dir) / "Fig20_joint_tradeoff_cost_vs_acc.png"
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved figure: {out}")


def main():
    base = load_config("configs/base.yaml")
    plot_cfg = load_config("configs/plotting.yaml")

    fig_dir = ensure_dir(plot_cfg["outputs"]["figures_dir"])
    dpi = int(plot_cfg["export"]["dpi"])

    rows = build_summary(plot_cfg)
    export_summary(plot_cfg, rows)

    if plot_cfg["figures"]["fig17_dynamic_tradeoff"]:
        fig17_dynamic_tradeoff(fig_dir, rows, dpi=dpi)

    if plot_cfg["figures"]["fig18_feature_privacy_tradeoff"]:
        fig18_feature_privacy_tradeoff(fig_dir, rows, dpi=dpi)

    if plot_cfg["figures"]["fig19_federated_tradeoff"]:
        fig19_federated_tradeoff(fig_dir, rows, dpi=dpi)

    if plot_cfg["figures"]["fig20_joint_pareto"]:
        fig20_joint_pareto(fig_dir, rows, plot_cfg, dpi=dpi)

    print("[INFO] Plotting finished")


if __name__ == "__main__":
    main()
