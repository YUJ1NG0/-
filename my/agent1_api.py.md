# agent_api.py

from flask import Flask, request, jsonify
from datetime import datetime
import redis
from sqlalchemy import create_engine, Column, String, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import threading
import time

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

# === 数据表模型 ===
class SensitiveWord(Base):
    __tablename__ = 'sensitive_words'
    id = Column(Integer, primary_key=True, autoincrement=True)
    word = Column(String(255), unique=True)
    source = Column(String(50))
    status = Column(String(50))
    created_at = Column(DateTime)

Base.metadata.create_all(engine)

# === API: 接收反馈词 ===
@app.route('/feedback', methods=['POST'])
def receive_feedback():
    data = request.json
    word = data.get('word')
    if not word:
        return jsonify({'error': 'word field is required'}), 400
    redis_client.sadd(REDIS_SET_KEY, word)
    return jsonify({'message': f'Word {word} added to Redis'}), 200

# === 后台任务：定期持久化 Redis 到 MySQL ===
def persist_to_mysql(interval=60):
    while True:
        try:
            session = Session()
            words = redis_client.smembers(REDIS_SET_KEY)
            for word in words:
                exists = session.query(SensitiveWord).filter_by(word=word).first()
                if not exists:
                    new_word = SensitiveWord(
                        word=word,
                        source='periodic_flush',
                        status='active',
                        created_at=datetime.now()
                    )
                    session.add(new_word)
                    print(f'✅ 持久化到 MySQL: {word}')
            session.commit()
            session.close()
        except Exception as e:
            print(f'❌ 持久化错误: {e}')
        time.sleep(interval)

# === 启动后台线程和 Flask 应用 ===
if __name__ == '__main__':
    threading.Thread(target=persist_to_mysql, daemon=True).start()
    app.run(host='0.0.0.0', port=5001)