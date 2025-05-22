以下是 ✅ **完整代码片段**，用于接入本地大模型（如 DeepSeek）生成敏感词变体，并将其自动写入 Redis 和数据库，形成完整闭环流程。

------

## 🧩 模块结构

```
1. 接收词 → Redis (set)
2. 定时任务 → Redis → 检查是否新词 → 写入 MySQL → 写入 Faiss
3. 使用本地大模型生成变体
4. 变体经过语义 + 拼写距离过滤
5. 合格变体自动回写 Redis（再次进入流程）
```

------

## ✅ 完整 Python 代码（关键部分整合）

你只需将这些函数集成到你的调度模块里。

### 🔁 1. Redis 拉词并检查新词

```python
def sync_from_redis_to_mysql_faiss():
    words = list(redis_client.smembers(REDIS_SET_KEY))
    if not words:
        print("🔍 No new words in Redis.")
        return

    for word in words:
        if is_duplicate_in_db(word):
            redis_client.srem(REDIS_SET_KEY, word)
            continue
        add_sensitive_word(word, source="feedback")
        add_to_faiss(word)
        redis_client.srem(REDIS_SET_KEY, word)
        generate_and_insert_variants(word)
```

------

### 🧠 2. 生成变体（本地大模型）

```python
def generate_variants_with_deepseek(word: str) -> list[str]:
    prompt = f"""请为敏感词“{word}”生成 5 个常见的拼写或符号变体，输出格式仅为：
z*i*y*、ziyou、z.y.o.u、自☆由、z_y_o_u
现在请生成“{word}”的变体："""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    parts = result.split("：")
    raw = parts[-1] if len(parts) > 1 else parts[0]
    return [x.strip() for x in raw.replace("、", ",").split(",") if x.strip()]
```

------

### 🧠 3. 相似度过滤 + Redis 回写

```python
def generate_and_insert_variants(base_word: str):
    variants = generate_variants_with_deepseek(base_word)
    for var in variants:
        if is_valid_variant(base_word, var) and not is_duplicate_in_db(var):
            redis_client.sadd(REDIS_SET_KEY, var)
            print(f"🧬 加入 Redis（新变体）: {var}")
```

------

### ✅ 4. 是否为已有词（DB + Faiss 检查）

```python
def is_duplicate_in_db(word: str) -> bool:
    session = Session()
    exists = session.query(SensitiveWord).filter_by(word=word).first()
    session.close()
    return exists is not None

def is_similar_to_existing(word: str, threshold=0.6):
    global faiss_index, word_list
    if not word_list or not faiss_index:
        return False
    emb = embedding_model.encode(word, convert_to_tensor=True).cpu().numpy().reshape(1, -1)
    scores, indices = faiss_index.search(emb, k=1)
    return scores[0][0] >= threshold
```

------

### 🔁 5. 添加到 Faiss

```python
def add_to_faiss(word: str):
    global faiss_index, word_list
    vector = embedding_model.encode(word).reshape(1, -1).astype("float32")
    faiss_index.add(vector)
    word_list.append(word)
    faiss.write_index(faiss_index, "faiss_index.idx")
    with open("word_list.json", "w", encoding="utf-8") as f:
        json.dump(word_list, f)
```

------

## 🕓 定时调度器入口（每 60s）

```python
import time

if __name__ == '__main__':
    load_faiss_index()
    while True:
        sync_from_redis_to_mysql_faiss()
        time.sleep(60)
```

------

## 📌 系统总览（闭环）

```
            用户反馈/检测生成词
                      ↓
                Redis: feedback
                      ↓
     定时任务读取 Redis + 持久化到 MySQL
                      ↓
     → 添加到 Faiss          ↘
                      生成变体（大模型）
                          ↓
              过滤后再次写入 Redis（循环）
```

------

是否需要我为你：

- 将以上流程打包为服务（Flask/FastAPI）？
- 构建一个 CLI 工具可手动测试某个词是否重复？
- 增加关键词管理 Web UI（前端页面）？

只需告诉我，立即为你继续整合。