import requests
from PIL import Image
import io
import time
import concurrent.futures
import statistics


def run_test():
    img1 = Image.new("RGB", (128, 128), "white")
    img2 = Image.new("RGB", (64, 32), "black")

    img_bytes1 = io.BytesIO()
    img_bytes2 = io.BytesIO()
    img1.save(img_bytes1, format="PNG")
    img2.save(img_bytes2, format="PNG")

    try:
        start_time = time.time()
        response = requests.post(
            "http://localhost:11111/process",
            files=[
                ("images", ("white.png", img_bytes1.getvalue(), "image/png")),
                ("images", ("black.png", img_bytes2.getvalue(), "image/png"))
            ],
            data={"queries": [
                "Is attention really all you need?",
                "What is the amount of bananas farmed in Salvador?"
            ]},
            timeout=30  # 设置超时时间为30秒
        )
        elapsed_time = time.time() - start_time

        if response.status_code == 200:
            return (True, elapsed_time, response.status_code)
        else:
            return (False, elapsed_time, response.status_code)
    except Exception as e:
        return (False, 0, str(e))


def run_batch(batch_size=50):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_test) for _ in range(batch_size)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    return results


def main():
    total_tests = 100
    batch_size = 50
    batches = total_tests // batch_size

    success_count = 0
    failure_count = 0
    response_times = []
    status_codes = {}
    errors = {}

    print(f"Starting stress test: {batches} batches of {batch_size} requests each")

    for i in range(batches):
        print(f"Running batch {i + 1}/{batches}")
        batch_results = run_batch(batch_size)

        for success, elapsed_time, status in batch_results:
            if success:
                success_count += 1
                response_times.append(elapsed_time)
            else:
                failure_count += 1
                if isinstance(status, int):
                    status_codes[status] = status_codes.get(status, 0) + 1
                else:
                    errors[status] = errors.get(status, 0) + 1

        # 打印中间结果
        if response_times:
            avg_time = statistics.mean(response_times)
            print(f"  Success: {success_count}, Failures: {failure_count}, Avg response time: {avg_time:.4f}s")

    # 打印最终统计信息
    print("\nFinal Results:")
    print(f"Total requests: {total_tests}")
    print(f"Success: {success_count} ({success_count / total_tests * 100:.2f}%)")
    print(f"Failures: {failure_count} ({failure_count / total_tests * 100:.2f}%)")

    if response_times:
        print("\nResponse Time Statistics:")
        print(f"  Average: {statistics.mean(response_times):.4f}s")
        print(f"  Median: {statistics.median(response_times):.4f}s")
        print(f"  Min: {min(response_times):.4f}s")
        print(f"  Max: {max(response_times):.4f}s")
        print(f"  95th percentile: {statistics.quantiles(response_times, n=20)[18]:.4f}s")

    if status_codes:
        print("\nStatus Code Distribution:")
        for code, count in status_codes.items():
            print(f"  {code}: {count} times")

    if errors:
        print("\nError Distribution:")
        for error, count in errors.items():
            print(f"  {error}: {count} times")


if __name__ == "__main__":
    main()
