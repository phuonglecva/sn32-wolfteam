import hashlib
import traceback
import redis

redis_pool = redis.ConnectionPool(host='127.0.0.1', port=6379, decode_responses=True)


def get_conn():
    conn = redis.Redis(connection_pool=redis_pool)
    return conn


def exists(key, db=0):
    try:
        conn = get_conn()
        conn.select(db)
        return conn.exists(key) == 1
    except Exception as e:
        print(e)
        traceback.print_exc()


def get(key, db=0):
    try:
        conn = get_conn()
        conn.select(db)
        return conn.get(key)
    except Exception as e:
        print(e)
        traceback.print_exc()


def set(key, value, db=0):
    try:
        conn = get_conn()
        conn.select(db)
        conn.setex(key, 3600, value)
    except Exception as e:
        print(e)
        traceback.print_exc()


def gen_hash(token):
    m = hashlib.sha256(token.encode('UTF-8'))
    sha256_hex = m.hexdigest()
    return sha256_hex


def set_pred_result(texts, preds, db=0):
    print(f'start set redis len(texts) = {len(texts)}, db = {db}')
    try:
        if len(texts) != len(preds):
            return None
        conn = get_conn()
        conn.select(db)
        hashes = [gen_hash(text) for text in texts]
        for i in range(len(hashes)):
            key = hashes[i]
            value = str(preds[i])
            conn.setex(key, 3600, value)
    except Exception as e:
        print(e)
        traceback.print_exc()


def get_pred_result(texts, db=0):
    print(f'start get redis len(texts) = {len(texts)}, db = {db}')
    try:
        result = [None] * len(texts)
        conn = get_conn()
        conn.select(db)
        hashes = [gen_hash(text) for text in texts]
        for i in range(len(hashes)):
            key = hashes[i]
            value = conn.get(key)
            if value is not None:
                result[i] = (value == 'True')
        return result
    except Exception as e:
        print(e)
        print(f'get_pred_result length of texts: {len(texts)}')
        traceback.print_exc()
        return [None] * len(texts)


if __name__ == '__main__':
    print(gen_hash("abc"))
    # set('abc', '123')
    # val = get('abc')
    # print(val)
    # e = exists('abc')
    # print(e)
