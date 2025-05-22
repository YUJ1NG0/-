# agent_api.py

from flask import Flask, request, jsonify
from datetime import datetime
import redis
from sqlalchemy import create_engine, Column, String, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import threading
import time
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# === 配置 ===
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_SET_KEY = 'sensitive:words:set'
MYSQL_URI = 'mysql+pymysql://root:password@localhost:3306/sensitive_db'

# === 初始化组件 ===
app = Flask(__name__)
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
Base = declarative_base()
engine = create_engine(MYSQL_URI)
Session = sessionmaker(bind=engine)
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# === 数据表模型 ===
class SensitiveWord(Base):
    __tablename__ = 'sensitive_words'
    id = Column(Integer, primary_key=True, autoincrement=True)
    word = Column(String(255), unique=True)
    source = Column(String(50))
    status = Column(String(50))
    created_at = Column(DateTime)

Base.metadata.create_all(engine)

# === 初始化向量数据库（Faiss） ===
DIM = 384  # for MiniLM
index = faiss.IndexFlatL2(DIM)
word_list = []  # 保留与向量顺序一致的词

def rebuild_faiss_index():
    global word_list, index
    session = Session()
    words = session.query(SensitiveWord.word).all()
    session.close()
    word_list = [w[0] for w in words]
    if word_list:
        embeddings = embedding_model.encode(word_list, convert_to_numpy=True)
        index.reset()
        index.add(embeddings)
        print(f"🔍 向量索引已重建，共 {len(word_list)} 个词")

rebuild_faiss_index()

# === API: 接收反馈词 ===
@app.route('/feedback', methods=['POST'])
def receive_feedback():
    data = request.json
    word = data.get('word')
    if not word:
        return jsonify({'error': 'word field is required'}), 400
    redis_client.sadd(REDIS_SET_KEY, word)
    return jsonify({'message': f'Word {word} added to Redis'}), 200

# === 相似词查询（用向量数据库加速） ===
def is_similar_to_existing(word: str, threshold=0.6) -> bool:
    if not word_list:
        return False
    try:
        embedding = embedding_model.encode([word], convert_to_numpy=True)
        D, I = index.search(embedding, k=1)
        sim_score = 1 - D[0][0] / 4  # L2 距离转相似度估算（MiniLM 余弦近似）
        return sim_score >= threshold
    except Exception as e:
        print(f"❌ 相似度查询错误: {e}")
        return False

# === 后台任务：定期持久化 Redis 到 MySQL ===
def persist_to_mysql(interval=60):
    while True:
        try:
            session = Session()
            words = redis_client.smembers(REDIS_SET_KEY)
            inserted = 0
            for word in words:
                exists = session.query(SensitiveWord).filter_by(word=word).first()
                if not exists:
                    if not is_similar_to_existing(word):
                        new_word = SensitiveWord(
                            word=word,
                            source='periodic_flush',
                            status='active',
                            created_at=datetime.now()
                        )
                        session.add(new_word)
                        inserted += 1
            session.commit()
            session.close()
            if inserted:
                rebuild_faiss_index()
        except Exception as e:
            print(f'❌ 持久化错误: {e}')
        time.sleep(interval)

# === 启动后台线程和 Flask 应用 ===
if __name__ == '__main__':
    threading.Thread(target=persist_to_mysql, daemon=True).start()
    app.run(host='0.0.0.0', port=5001)