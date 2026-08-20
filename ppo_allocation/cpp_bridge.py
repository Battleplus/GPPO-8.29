import argparse
import contextlib
import io
import json
import sys
from pathlib import Path


def _write_tasks_tsv(path: Path, response: dict) -> None:
    result = response.get("result") or {}
    uav_tasks = result.get("uav_tasks") or {}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("uav_id\talive\ttask\tsensor\tregions\ttarget_id\n")
        for uav_id in sorted(uav_tasks):
            detail = uav_tasks[uav_id] or {}
            regions = detail.get("regions") or []
            if isinstance(regions, list):
                region_text = ",".join(str(item) for item in regions)
            else:
                region_text = str(regions)
            f.write(
                "\t".join([
                    str(uav_id),
                    "1" if bool(detail.get("alive", False)) else "0",
                    str(detail.get("task", "")),
                    str(detail.get("sensor", "")),
                    region_text,
                    "" if detail.get("target_id") is None else str(detail.get("target_id", "")),
                ])
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="C++/外部进程可调用的任务重分配桥接脚本")
    parser.add_argument("--request-file", required=True, help="包含请求 JSON 的文件路径")
    args = parser.parse_args()

    request_path = Path(args.request_file)
    if not request_path.exists():
        raise FileNotFoundError(f"请求文件不存在: {request_path}")

    request = json.loads(request_path.read_text(encoding="utf-8"))
    stdout_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stdout_buffer):
        ppo_dir = Path(__file__).resolve().parent
        if str(ppo_dir) not in sys.path:
            sys.path.insert(0, str(ppo_dir))
        from reallocation_service import reallocate_cpp_interface
        response = reallocate_cpp_interface(request)
    tasks_output_path = request.get("tasks_output_path")
    if tasks_output_path and response.get("success"):
        _write_tasks_tsv(Path(tasks_output_path), response)
    print(json.dumps(response, ensure_ascii=False, indent=2), file=sys.stdout)


if __name__ == "__main__":
    main()
