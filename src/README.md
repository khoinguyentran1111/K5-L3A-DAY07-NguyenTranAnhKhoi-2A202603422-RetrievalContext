# Code Day 7 — Agentic RAG trên vector store Day 6

Code nằm trong [src/day7/](day7/). Starter gốc vẫn giữ nguyên trong [starter/](../starter/) để tham khảo.

| File | Việc làm |
|---|---|
| [config.py](day7/config.py) | Đường dẫn, biến môi trường, quy tắc chọn cấu hình được viết **trước** khi chạy dev |
| [clients.py](day7/clients.py) | Client OpenRouter cho embedding và chat; retry, đếm token và chi phí (`usage.cost`), không có mock fallback |
| [prepare_data.py](day7/prepare_data.py) | Adapter: chunk B và benchmark Day 6 → schema Day 7; đối chiếu id + hash với pgvector |
| [data.py](day7/data.py) | Đọc dữ liệu và **một luật hit chung** cho mọi phương pháp |
| [retrievers.py](day7/retrievers.py) | Dense (pgvector), BM25, Hybrid RRF, Query rewrite, LLM rerank |
| [pipelines.py](day7/pipelines.py) | Đăng ký pipeline theo tên, dùng chung cho benchmark và agent |
| [context.py](day7/context.py) | Chọn context: dedup, giữ thứ hạng, mở rộng đoạn lân cận, ngân sách ký tự |
| [eval_retrieval.py](day7/eval_retrieval.py) | Benchmark retrieval/context: Hit@1, Evidence Hit@3, Hit@5, MRR@10, Context Evidence Rate |
| [aggregate_repeats.py](day7/aggregate_repeats.py) | Trung bình và min–max qua các lần chạy lặp |
| [agent.py](day7/agent.py) | Agent tool calling, cấu hình A0/A1/A2, kiểm tra citation, từ chối |
| [run_agent.py](day7/run_agent.py) | Chạy agent trên một split; lưu trace, metric tự động và mẫu nhãn chấm tay |
| [write_reviews.py](day7/write_reviews.py) | Chuyển nhãn tay (`labels.json`) thành `reviews.json` theo thứ tự citation |
| [agent_report.py](day7/agent_report.py) | Gộp trace và nhãn tay sang định dạng của `starter/agent_metrics.py` |

## 1. Cài đặt

Cần Python 3.11+ và repo Day 6 đặt cùng thư mục cha (hoặc đặt biến `DAY6_ROOT`). Vector store là container PostgreSQL + pgvector của Day 6.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install "psycopg[binary]==3.3.8" "openai==3.50.0" "python-dotenv>=1,<2"
copy .env.example .env      # rồi điền DATABASE_URL (mật khẩu theo compose Day 6) và OPENROUTER_API_KEY; không commit .env
docker start day6_student_starter-postgres-1     # hoặc: cd <Day6>/starter; docker compose -p day6_student_starter up -d --wait
```

Nếu volume Day 6 đã mất, tạo lại index bằng `python -m src.run index` trong `<Day6>/starter` (chi phí khoảng $0.0015).

## 2. Chạy

Chạy từ thư mục gốc repo. Trên Windows nên đặt `PYTHONIOENCODING=utf-8` để in được tiếng Việt.

```powershell
.venv\Scripts\python -m unittest discover -s tests -v                  # test offline (mock chỉ để test code)
.venv\Scripts\python -m src.day7.prepare_data --check-db                 # data/corpus, data/benchmark

# Retrieval trên dev (câu hỏi gốc + filter do người dùng cung cấp)
.venv\Scripts\python -m src.day7.eval_retrieval --split dev --pipelines dense bm25 hybrid_rrf rewrite_bm25 rewrite_dense rewrite_hybrid rewrite_hybrid_rerank --output results/retrieval/dev
.venv\Scripts\python -m src.day7.eval_retrieval --split dev --pipelines dense rewrite_dense rewrite_hybrid_rerank --budget 5000 --expand-top 2 --output results/retrieval/dev_neighbours

# Lặp lại các pipeline có LLM (LLM không tất định) và gộp trung bình [min–max]
.venv\Scripts\python -m src.day7.eval_retrieval --split dev --pipelines rewrite_bm25 rewrite_dense rewrite_hybrid rewrite_hybrid_rerank --output results/retrieval/dev_repeat2
.venv\Scripts\python -m src.day7.eval_retrieval --split dev --pipelines rewrite_bm25 rewrite_dense rewrite_hybrid rewrite_hybrid_rerank --output results/retrieval/dev_repeat3
.venv\Scripts\python -m src.day7.aggregate_repeats results/retrieval/dev results/retrieval/dev_repeat2 results/retrieval/dev_repeat3 --output results/retrieval/dev_repeats.md

# Agent trên dev (3 cấu hình), chấm tay baseline + cấu hình chọn
.venv\Scripts\python -m src.day7.run_agent --split dev --config A0_dense --output results/agent/dev/A0_dense
.venv\Scripts\python -m src.day7.run_agent --split dev --config A1_dense_neighbours --output results/agent/dev/A1_dense_neighbours
.venv\Scripts\python -m src.day7.run_agent --split dev --config A2_hybrid_rerank_neighbours --output results/agent/dev/A2_hybrid_rerank_neighbours
.venv\Scripts\python -m src.day7.write_reviews results/agent/dev/A0_dense results/agent/dev/A0_dense/labels.json
.venv\Scripts\python -m src.day7.write_reviews results/agent/dev/A1_dense_neighbours results/agent/dev/A1_dense_neighbours/labels.json
.venv\Scripts\python -m src.day7.agent_report results/agent/dev/A0_dense results/agent/dev/A1_dense_neighbours

# ---- Chốt cấu hình (results/freeze.json) TRƯỚC các lệnh test dưới đây ----
# Retrieval trên test: baseline + pipeline đã chọn (+ neighbours)
.venv\Scripts\python -m src.day7.eval_retrieval --split test --pipelines dense rewrite_hybrid_rerank --output results/retrieval/test
.venv\Scripts\python -m src.day7.eval_retrieval --split test --pipelines dense --budget 5000 --expand-top 2 --output results/retrieval/test_neighbours

# Agent trên test: baseline A0 + cấu hình chọn A1, rồi chấm tay và tổng hợp
.venv\Scripts\python -m src.day7.run_agent --split test --config A0_dense --output results/agent/test/A0_dense
.venv\Scripts\python -m src.day7.run_agent --split test --config A1_dense_neighbours --output results/agent/test/A1_dense_neighbours
.venv\Scripts\python -m src.day7.write_reviews results/agent/test/A0_dense results/agent/test/A0_dense/labels.json
.venv\Scripts\python -m src.day7.write_reviews results/agent/test/A1_dense_neighbours results/agent/test/A1_dense_neighbours/labels.json
.venv\Scripts\python -m src.day7.agent_report results/agent/test/A0_dense results/agent/test/A1_dense_neighbours
```

`labels.json` là nhãn chấm tay (soạn sau khi đọc trace), `write_reviews` chuyển chúng sang `reviews.json`; chạy lại agent sẽ ra câu trả lời khác nên phải chấm lại. Kết quả chi tiết và phân tích ở [báo cáo](../report/individual/2A202603422_NguyenTranAnhKhoi.md).
