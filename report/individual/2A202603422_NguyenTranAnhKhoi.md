# Báo cáo Day 7 — Agentic RAG

**Nguyễn Trần Anh Khôi · 2A202603422 · L3A.** Code: [src/day7/](../../src/day7/) · Cách chạy: [src/README.md](../../src/README.md) · Kết quả: [results/](../../results/)

> **Về nhãn chấm tay.** AI (Claude) soạn mọi nhãn (grounded answer, citation, abstention, context 0/1/2) sau khi đọc câu trả lời, expected answer, gold và **toàn văn** chunk được trích; tôi (NguyenTranAnhKhoi-2A202603422) đã đọc lại và xác nhận. Lý do của từng nhãn nằm trong các file `labels.json`.

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk.** OpenStax *College Success* (Amy Baldwin et al.), chương 3 *Managing Your Time and Priorities* và chương 6 *Studying, Memory, and Test Taking*, tương ứng trang PDF 75–112 và 181–205. Giấy phép CC BY-NC-SA 4.0; bản gốc tải ở <https://openstax.org/details/books/college-success>. Corpus dùng lại nguyên kết quả ingest của Day 6: 63 trang, trong đó 61 trang có nội dung. Index gồm **299 chunk** của chiến lược B (theo heading và đoạn văn, tối đa 800 ký tự; độ dài 56–859 ký tự, trung vị 583), 167 chunk ở chương 3 và 132 chunk ở chương 6. Có 37 chunk vắt qua hai trang, nên metadata lưu thêm `page_end`. Adapter [prepare_data.py](../../src/day7/prepare_data.py) chuyển dữ liệu sang schema Day 7 ([data/corpus/chunks.json](../../data/corpus/chunks.json)) và đối chiếu với pgvector: 299/299 chunk khớp cả id lẫn SHA-256 của text, không thừa, không thiếu ([prepare_report.json](../../data/benchmark/prepare_report.json)).
- **Vector store và embedding.** PostgreSQL 16.16 + pgvector 0.8.7 (container Docker của Day 6, bảng `chunks`, `strategy='B'`), tìm kiếm chính xác bằng khoảng cách cosine `<=>`. Embedding model là `openai/text-embedding-3-small` qua OpenRouter, **1536 chiều**. Query embedding dùng đúng model và đúng lời gọi như lúc index Day 6, và code kiểm tra số chiều trước khi truy vấn. Không tạo index mới. Dense trên dev cho Hit@1 14/20 và Hit@3 18/20, trùng với kết quả strategy B của Day 6, xác nhận kết nối đúng.
- **Agent model, cách chạy, phân chia dev/test.** Agent dùng `anthropic/claude-haiku-5.5` qua OpenRouter (OpenAI-compatible tool calling), temperature 0, tắt reasoning (lý do ở phần lỗi). Benchmark gồm **31 câu**: dev có 20 câu có đáp án và 3 câu thiếu bằng chứng; test có 5 câu có đáp án và 3 câu thiếu bằng chứng ([cases.json](../../data/benchmark/cases.json)). 25 câu có đáp án lấy từ benchmark Day 6 với split giữ nguyên, đã đóng băng ngày 2026-11-13. Ở Day 6, test chỉ được chạy sau khi chọn chiến lược bằng quy tắc dev; ở Day 7, test **không được dùng để chỉnh**. Thời điểm chốt cấu hình được ghi trong [results/freeze.json](../../results/freeze.json) (2026-11-14T15:07:07+07:00), trước mọi lệnh chạy test. Giới hạn: dev và test cùng chủ đề và chương, nên test không tách biệt về chủ đề.
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm.**
  - **Câu có đáp án:** `gold` gồm evidence chính và các evidence thay thế của Day 6, mỗi evidence là câu trích nguyên văn tiếng Anh nằm trọn trong một chunk đúng nguồn và trang (25/25 câu đạt, không phải bỏ evidence nào). Câu hỏi viết bằng tiếng Việt, còn corpus là tiếng Anh. Các loại câu: tra cứu trực tiếp, diễn đạt lại, điều kiện, và câu có filter `chapter`/`section` (filter mô phỏng việc người dùng chọn chương trước khi hỏi, không suy ra từ gold).
  - **Câu thiếu bằng chứng (u01–u06, [unanswerable.json](../../data/benchmark/unanswerable.json)):** đều là câu gần chủ đề, ví dụ năm ra đời của Pomodoro (sách có nhắc Cirillo nhưng không có năm), hay GPA để giữ học bổng (sách chỉ có ví dụ GPA 3.0 trong mục SMART). Mỗi câu ghi `absence_reason` và `check_method`. Sự vắng mặt được kiểm tra bằng từ khóa trên toàn bộ 61 trang text của Day 6 (không chỉ dựa vào top-k), sau đó đọc các trang liên quan.
  - **Luật hit chung** ([data.py](../../src/day7/data.py)): một chunk là hit khi đúng `source_id`, khoảng trang của chunk chứa trang gold, và chunk chứa trọn một chuỗi evidence (sau khi chuẩn hóa khoảng trắng và dấu nháy). Câu thiếu bằng chứng không tính vào mẫu số của retrieval metrics.

## Retrieval và thử nghiệm

Năm phương pháp được triển khai trong [retrievers.py](../../src/day7/retrievers.py), tất cả trên cùng corpus, cùng index và cùng nhãn:
1. **Dense:** pgvector của Day 6.
2. **BM25:** tự cài đặt, k1=1.5, b=0.75, tokenizer tiếng Anh, bỏ stopword, stemming nhẹ.
3. **Hybrid RRF:** k=60, gộp top-20 của dense và top-20 của BM25, xếp theo thứ hạng nên không phải cộng score khác thang.
4. **Query rewrite:** LLM viết lại câu hỏi thành query tiếng Anh, rồi chạy BM25, dense hoặc hybrid.
5. **LLM rerank:** lấy top-15 từ hybrid, rồi LLM xếp lại theo câu hỏi gốc.

Mọi kết quả đều có id, score, rank, nguồn, trang và section. Filter là whitelist (`chapter`, `section`) và truyền tham số vào SQL. Lỗi của pgvector hoặc provider được ghi vào trace, không bị thay bằng kết quả giả.

Benchmark retrieval: lấy top-10 bằng **câu hỏi gốc** cộng filter người dùng cung cấp. Context là top-3 đóng gói trong ngân sách **3000 ký tự** (tính cả header nguồn/trang); chunk không vừa thì bỏ hẳn, không cắt. Riêng cấu hình đoạn lân cận dùng 5000 ký tự. Thời gian là trung vị mỗi query; chi phí là USD OpenRouter (`usage.cost`) cho toàn bộ 20 query.

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | Dense pgvector, top-3 vào context, filter người dùng, 3000 ký tự | Phương pháp gốc của Day 6 | dev | 14/20 (70%) | 18/20 (90%) | 18/20 (90%) | 0.9 s · $0.00002 |
| Phương pháp 2 | BM25 trên câu hỏi gốc tiếng Việt | Lexical; dự đoán yếu vì lệch ngôn ngữ VN↔EN | dev | 4/20 (20%) | 7/20 (35%) | 7/20 (35%) | ~0 s · $0 |
| Phương pháp 3 | Hybrid RRF (dense + BM25) | Gộp hai tín hiệu | dev | 12/20 (60%) | 18/20 (90%) | 18/20 (90%) | 0.9 s · $0.00002 |
| Cải thiện 1 (cũng là phương pháp 4) | Query rewrite v2 sang tiếng Anh → hybrid RRF | Lỗi q21: lệch ngôn ngữ; query tiếng Anh sẽ đưa evidence lên cao hơn | dev | 15.0/20 [14–16] | 19.0/20 [19–19] | 19.0/20 | 2.1 s · $0.00077 |
| Cải thiện 2 (cũng là phương pháp 5) | Rewrite v2 → hybrid top-15 → LLM rerank | Rerank theo câu hỏi gốc sẽ đưa evidence lên hạng 1 | dev | 16.7/20 [16–18] | **19.3/20 [19–20]** | 19.3/20 | 3.4 s · $0.0075 |
| Cải thiện 3 | Dense + thêm chunk liền trước/sau của top-2, 5000 ký tự | Lỗi q06: đáp án nằm ở chunk liền sau (lỗi ranh giới chunk Day 6) | dev | 14/20 | 18/20 | **19/20 (95%)** | 0.9 s · $0.00002 · 3737 ký tự |
| Cấu hình đã chọn | Rewrite v2 → hybrid → LLM rerank | Đã chốt trước test | test | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) | 3.5 s · $0.0019 |
| Baseline | Dense | | test | 2/5 (40%) | 4/5 (80%) | 4/5 (80%) | 1.0 s · $0.00001 |

Các cấu hình có LLM được chạy **3 lần** trên dev (bảng ghi trung bình [min–max]), vì dù temperature = 0 thứ hạng vẫn thay đổi giữa các lần chạy, chênh tới 2 câu Hit@1 ([dev_repeats.md](../../results/retrieval/dev_repeats.md)). Bảng đầy đủ, gồm cả Hit@5, MRR@10, `rewrite_bm25` và `rewrite_dense`, nằm ở [results/retrieval/dev/summary.md](../../results/retrieval/dev/summary.md). Một số đáng chú ý: `rewrite_bm25` đạt Hit@3 17.3/20 so với BM25 trên câu gốc 7/20, cho thấy BM25 yếu chủ yếu vì lệch ngôn ngữ. `rewrite_dense` đạt 19.0/20 và MRR 0.879, gần bằng rerank với chi phí bằng khoảng 1/10. Chạy đoạn lân cận trên test: [results/retrieval/test_neighbours/](../../results/retrieval/test_neighbours/).

**Phân tích lỗi trên dev** (trace ở [results/retrieval/dev/](../../results/retrieval/dev/) và [results/agent/](../../results/agent/)):

1. **q06: lỗi ranh giới chunk.** Câu hỏi: nếu không kịp làm cả hai bài quan trọng và gấp thì làm gì trước khi bỏ một bài. Dense lấy được chunk mở đầu ví dụ (`ch3:048654-049439`, hạng 2), nhưng câu trả lời ("the first step is to try to find a way to get everything finished… communicate with your instructors") nằm ở **chunk liền sau** (`ch3:049440-050104`, hạng 5), vì chunking của Day 6 cắt đôi ví dụ. Rewrite còn làm evidence rơi khỏi top-10. **Thử:** thêm chunk lân cận của top-2 vào context (Cải thiện 3, cấu hình agent A1). **Kết quả:** Context Evidence Rate của retrieval tăng từ 18 lên 19/20. Với agent, A0 trả lời q06 sai (0 điểm) còn A1 trích đúng chunk lân cận (2 điểm). Đổi lại, context dài hơn khoảng 1.8 lần.
2. **q21: lệch ngôn ngữ VN↔EN.** "Để tạo môi trường làm việc ít bị gián đoạn…": dense trên câu tiếng Việt chỉ đưa evidence lên hạng 4. Query tiếng Anh ("distraction-free study environment, minimize interruptions") đưa nó lên hạng 1. Đây là lý do chính để thêm query rewrite, và cũng giải thích BM25 trên câu gốc đạt 7/20.
3. **Rewrite v1 rò rỉ kiến thức nền.** Prompt v1 cho phép thêm "từ khóa có thể có", và model đã chèn luôn đáp án đoán được, ví dụ q14 "10 minutes", q02 "two hours per hour in class". Đây không phải gold, nhưng làm metric đẹp hơn thực tế. Prompt v2 cấm thêm dữ kiện, con số hoặc tên không có trong câu hỏi. Kết quả v2 không giảm mà còn tăng (Hit@3 của rewrite_hybrid từ 18 lên 19/20). Kết quả v1 lưu ở [archive/dev_rewrite_v1](../../results/retrieval/archive/dev_rewrite_v1/). Tuy vậy rò rỉ vẫn còn: trên test, rewrite v2 vẫn thêm "7 plus or minus 2" cho q12.
4. **Lỗi kỹ thuật: content rỗng.** Ở lần chạy đầu, 7/20 rewrite trả về chuỗi rỗng vì model mặc định dùng reasoning tokens, chiếm hết `max_tokens=120`. Lỗi được ghi vào trace với `errors=7` chứ không bị che. Sửa bằng cách tắt reasoning và báo lỗi rõ khi content rỗng.

**Chọn cấu hình.** Quy tắc được viết trong [config.py](../../src/day7/config.py) trước khi chạy dev:
- **Retrieval:** Evidence Hit@3 cao nhất trên dev; nếu hòa thì Hit@1; rồi đến chi phí và độ trễ. Chọn `rewrite_hybrid_rerank` (Hit@3 trung bình 19.33/20). Mức chênh 0.33 câu so với `rewrite_dense` nằm trong biên độ nhiễu giữa các lần chạy, nên nếu ưu tiên chi phí thì `rewrite_dense` là lựa chọn hợp lý.
- **Agent:** chọn theo quy tắc riêng, xem phần dưới.

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi** ([agent.py](../../src/day7/agent.py)):
  - **Tool:** một tool `search_documents(query)`. Model tự viết query (tiếng Anh) và quyết định có tìm thêm hay không. Filter do người dùng cung cấp được áp dụng tự động và ghi vào trace.
  - **Kết quả mỗi lượt:** top-3 chunk từ pipeline của cấu hình, cộng chunk lân cận nếu cấu hình có bật.
  - **Chọn context:** dedup theo id (chunk đã xem chỉ được nhắc lại bằng alias); ngân sách **8000 ký tự cho mỗi câu hỏi**, cộng dồn qua các lượt; chunk được gắn alias ngắn `[C1]`, `[C2]`… để model trích dẫn chính xác, và trace lưu bảng ánh xạ alias → chunk ID.
  - **Điều kiện dừng:** model trả lời (không gọi tool nữa); hoặc đạt `max_searches=4` thì buộc trả lời với `tool_choice=none`; hoặc đạt `max_turns=7`.
  - **Kiểm tra câu trả lời cuối:** câu trả lời là JSON `answered | insufficient_evidence`. Citation phải là alias đã xuất hiện trong context **và** phải có mặt trong nội dung câu trả lời. Nếu không đạt, model có 1 lượt sửa (không có tool); vẫn sai thì chuyển thành từ chối, và câu trả lời gốc được giữ trong `raw_final`.
  - **An toàn và lỗi:** system prompt nói rõ nội dung tài liệu là dữ liệu, không phải chỉ dẫn. Lỗi provider hoặc vector store được ghi `status: error` và không có câu trả lời giả; thực tế không có lỗi nào trong toàn bộ các lần chạy chính thức.
  - **Thực tế số lượt gọi:** model dùng 1–2 lượt search mỗi câu, chưa câu nào chạm giới hạn 4.
- **Các cấu hình agent và quy tắc chọn đăng ký trước.** Điểm chọn = số câu có evidence trong context + số câu thiếu bằng chứng được từ chối. Nếu hòa thì so số lần từ chối nhầm, rồi đến chi phí.
  - **A0_dense (baseline):** dense.
  - **A1_dense_neighbours:** dense + chunk lân cận của top-2 (giả thuyết từ lỗi q06).
  - **A2_hybrid_rerank_neighbours:** hybrid + LLM rerank + chunk lân cận.

  Kết quả dev: A0 = 19 + 3 = 22; A1 = 20 + 3 = 23; A2 = 20 + 3 = 23. A1 và A2 hòa điểm, hòa cả số lần từ chối nhầm (0), nên **chọn A1** vì chi phí chỉ khoảng một nửa A2 ($0.0133 so với $0.0258 cho 23 câu; A2 chậm hơn, 8.0 s so với 5.4 s). Chấm tay A0 và A1 trên dev trước khi chốt xác nhận lựa chọn: 1.95 so với 1.85/2. A2 không được chấm tay, chỉ có metric tự động ([summary](../../results/agent/dev/A2_hybrid_rerank_neighbours/summary.json)).
- **Lỗi agent trên dev, trước và sau khi sửa: từ chối nhầm vì định dạng citation.** Ở lần chạy A0 đầu tiên ([archive](../../results/agent/archive/dev_A0_dense_v1_no_citation_repair/)), q02 và q10 bị chuyển thành từ chối dù evidence đã nằm trong context. Model trả lời **đúng**, nhưng chỉ điền danh sách `citations` mà không chèn `[C#]` vào nội dung, nên bộ kiểm tra loại bỏ. Cách sửa: prompt yêu cầu bắt buộc chèn `[C#]` sau mỗi ý, và thêm 1 lượt sửa. Kết quả: từ chối nhầm trên dev giảm từ 2/20 xuống 0/20. Lượt sửa không phải kích hoạt lần nào ở các lần chạy sau, nên việc sửa prompt đã đủ; lượt sửa được giữ làm lưới an toàn.

Trong bảng dưới: grounded answer chỉ tính câu có đáp án; citation hợp lệ là số citation vừa nằm trong context vừa được đọc xác nhận là hỗ trợ ý tương ứng; "câu có citation" là số câu đã trả lời có kèm citation. Công cụ tổng hợp: [agent_report.py](../../src/day7/agent_report.py) gọi `starter/agent_metrics.py`.

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| A0_dense / dev | 23/23 | 1 lượt: 7 câu, 2 lượt: 16 câu | 1 / 1 / 18 · **1.85/2** | 38/39 | 20/20 | 3/3 | 0/20 |
| A1_dense_neighbours / dev | 23/23 | 1: 7, 2: 16 | 0 / 1 / 19 · **1.95/2** | 47/48 | 20/20 | 3/3 | 0/20 |
| A2_hybrid_rerank_neighbours / dev | 23/23 | 1: 8, 2: 15 | không chấm tay | — | — | 3/3 (tự động) | 0/20 |
| **A0_dense / test (baseline)** | 8/8 | 1: 1, 2: 7 | 1 / 0 / 4 · **1.60/2** | 10/10 | 4/4 | 3/3 | 1/5 |
| **A1_dense_neighbours / test (đã chọn)** | 8/8 | 2: 8 | 0 / 0 / 5 · **2.00/2** | 17/17 | 5/5 | 3/3 | 0/5 |

Context Evidence Rate của agent (evidence nằm trong context cuối gửi cho model):

| Cấu hình | Dev | Test |
|---|---|---|
| A0 | 19/20 | 4/5 |
| A1 | 20/20 | 5/5 |
| A2 | 20/20 | — |

Chấm context 0/1/2 thủ công cho 5 câu test có đáp án. Evidence của cả 5 câu đều là một câu tự đủ nghĩa, nên khi chunk chứa evidence có mặt kèm nguồn và trang thì chấm 2.

| Cấu hình | Mức 2 | Mức 1 | Mức 0 |
|---|---|---|---|
| A0 | 4 câu | 0 | 1 câu (q25) |
| A1 | 5 câu | 0 | 0 |
| Retrieval dense (top-3) | 4 câu | 0 | 1 câu (q04) |
| Retrieval rewrite_hybrid_rerank (top-3) | 5 câu | 0 | 0 |

**Từng câu test của cấu hình đã chọn (A1),** kèm so sánh với A0 khi khác nhau. Nhãn và lý do đầy đủ ở [A1 labels.json](../../results/agent/test/A1_dense_neighbours/labels.json) và [A0 labels.json](../../results/agent/test/A0_dense/labels.json).

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| q04 | Điều kiện | 2 lượt: "procrastination when does it become a serious problem" → evidence ở hạng 1 (`ch3:018281-018764`, trang 83) | Ba trường hợp: thói quen mãn tính; nhiều việc mà ít thời gian; việc rất quan trọng. Thêm các hậu quả [C1][C9][C10][C11][C4–C6] | 2. Đủ ba điều kiện; các ý thêm đều có nguồn. 7/7 citation đúng. A0 cũng đạt 2. |
| q12 | Trực tiếp | 2 lượt: "George Miller 1956 short-term memory…" → hạng 1 (trang 184) | Khoảng 5–9 đơn vị [C1] | 2. A0 cũng đạt 2. |
| q19 | Điều kiện | 2 lượt → hạng 1 (`ch6:061372-061824`, trang 202) | Bỏ qua câu khó, làm câu chắc chắn trước, quay lại sau [C1][C2][C3] | 2. A0 cũng đạt 2. |
| q24 | Filter chương 6 | 2 lượt. Lượt 1 "phone distractions during study session": top-3 không có evidence, nhưng chunk evidence `ch6:027462-027843` (trang 191) được đưa vào context ngay lượt này vì là chunk liền sau của kết quả hạng 2 (nhờ đoạn lân cận). Lượt 2 lấy được nó ở hạng 1 | Cất điện thoại khuất tầm mắt hoặc sang phòng khác, úp xuống là chưa đủ [C5] | 2. A0 cũng đạt 2, nhưng phải đến lượt search thứ 2 mới thấy evidence. |
| q25 | Filter chương 3 | 2 lượt: lượt 1 "dividing 24 hours…" không có evidence; lượt 2 model tự viết "Eight hours sleep, eight hours work, eight hours leisure" → tìm được quy tắc 8-8-8 (trang 109) | Quy tắc 8-8-8 [C2] | 2. **Lưu ý:** tìm được nhờ kiến thức nền của model trong query, không nhờ đoạn lân cận. **A0 đạt 0:** chỉ search 1 lần rồi từ chối nhầm, đây là ví dụ lỗi trên test. |
| u04 | Thiếu bằng chứng | 2 lượt về Eisenhower và WWII | Sách chỉ nói Eisenhower là tổng chỉ huy Đồng minh, không nêu chiến dịch | Từ chối đúng, không bịa Normandy. |
| u05 | Thiếu bằng chứng | 2 lượt về giờ dùng mạng xã hội | Không có số liệu | Từ chối đúng. |
| u06 | Thiếu bằng chứng | 2 lượt về GPA và học bổng | GPA 3.0 chỉ là ví dụ của mục tiêu SMART, không phải điều kiện học bổng | Từ chối đúng; không bị câu gần chủ đề đánh lừa. |

**Ví dụ để trình bày:**
- **Câu trả lời có bằng chứng:** A1, q06 (dev). Agent trích đúng câu ở trang 94 nhờ đoạn lân cận.
- **Câu thiếu bằng chứng:** u06 (test).
- **Lỗi:** A0, q25 (test), từ chối nhầm vì chỉ search một lần. Thêm lỗi q09 (dev, A1): gán ý "chia khối 25 phút" của Pomodoro cho "ăn con ếch" và trích sai đoạn C2.

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:**
  - **Trace agent:** `results/agent/<split>/<config>/traces.json`. Mỗi câu lưu từng tool call (query, filter, kết quả kèm score/rank, các bước nội bộ của pipeline, chunk mới, chunk lặp, chunk bị bỏ vì vượt ngân sách), token và chi phí của mỗi lượt model, context cuối, câu trả lời (gồm bản dùng alias và bản dùng chunk ID), citation, độ trễ và lỗi. Nhãn ở `labels.json`; `reviews.json` và `agent_metrics.json` nằm cùng thư mục.
  - **Trace retrieval:** `results/retrieval/<run>/<pipeline>.traces.json`.
  - **Lệnh chạy:**

    ```powershell
    .venv\Scripts\python -m src.day7.run_agent --split test --config A1_dense_neighbours --output results/agent/test/A1_dense_neighbours
    .venv\Scripts\python -m src.day7.write_reviews results/agent/test/A1_dense_neighbours results/agent/test/A1_dense_neighbours/labels.json
    .venv\Scripts\python -m src.day7.agent_report results/agent/test/A0_dense results/agent/test/A1_dense_neighbours
    ```

    Các lệnh còn lại ở [src/README.md](../../src/README.md).
  - **Thời gian và chi phí mỗi câu:**

    | Cấu hình | Thời gian (trung vị) | Chi phí / câu |
    |---|---|---|
    | A0 | 5.3 s | $0.00050 |
    | A1 | 5.4 s | $0.00058 |
    | A2 | 8.0 s | $0.00112 |

    Phần lớn là chi phí của model agent; embedding gần như bằng 0. Tổng chi phí API của mọi lần chạy có ghi nhận là khoảng **$0.12**. Lần chạy A1/A2 bị dừng giữa chừng (vì phải sửa code) và các lần chạy thử không được ghi chi phí, ước tính dưới $0.02.
- **Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:**
  - **Trước/sau trên test:** grounded answer tăng từ 1.60 lên 2.00/2, Context Evidence Rate từ 4/5 lên 5/5, từ chối nhầm từ 1/5 xuống 0/5. Đổi lại, chi phí tăng khoảng 20% và context tăng khoảng 1.8 lần (dev trung bình 2574 → 4853 ký tự). Với retrieval không có agent, cấu hình rewrite + hybrid + rerank tăng Hit@1 từ 2/5 lên 5/5, nhưng chậm hơn khoảng 3.5 lần và đắt hơn khoảng 200 lần (vẫn chỉ $0.0004/query).
  - **Đánh đổi:** agent tự viết query tiếng Anh nên đã bù được phần lớn lỗi lệch ngôn ngữ. Vì vậy, khi đứng trong agent, rerank (A2) không tốt hơn dense + đoạn lân cận (A1) theo tiêu chí đã chọn, mà lại đắt gấp đôi.
  - **Giới hạn:**
    1. Test chỉ có 8 câu (5 câu có đáp án), nên mỗi câu có đáp án tương đương 20 điểm phần trăm; khác biệt A0/A1 trên test thực chất chỉ đến từ 1 câu (q25), và câu đó A1 thắng nhờ query do model tự đoán chứ không nhờ cải thiện đã đề xuất.
    2. LLM không tất định dù temperature = 0; agent chỉ chạy 1 lần mỗi cấu hình.
    3. Model có kiến thức nền về các chủ đề phổ biến (Pomodoro, 8-8-8, Miller 7±2) và đưa chúng vào query; điều này làm Hit/Context Evidence cao hơn so với một corpus riêng tư.
    4. Nhãn do AI soạn, sau đó tôi đọc lại và xác nhận. Chỉ có một người chấm nên chưa đo được độ đồng thuận giữa các người chấm.
    5. Benchmark kế thừa từ Day 6, phần lớn mỗi câu chỉ có một evidence; chưa có câu multi-hop.
    6. Đoạn lân cận lấy theo thứ tự đọc trong cùng chương, có thể thêm đoạn không liên quan nếu chunk nằm ở ranh giới section.

## Thử thách nâng cao (nếu thực hiện)

Chưa thực hiện ở lần nộp này.
