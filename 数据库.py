import pandas as pd
import mysql.connector
from mysql.connector import Error
import numpy as np
import traceback

# ========== 数据库配置 ==========
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',
    'database': 'news_recommendation',
    'auth_plugin': 'mysql_native_password',
    'connection_timeout': 300,  # 连接超时（秒）
    'autocommit': False
}


def safe_int(val):
    """安全转换为 Python int，兼容 pandas/NumPy"""
    if pd.isna(val) or val is None:
        return 0
    if isinstance(val, (np.integer, int)):
        return int(val)
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return 0


def safe_str(val):
    """安全转换为 Python str"""
    if pd.isna(val) or val is None:
        return ''
    return str(val).strip()


def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)


def main():
    print("正在读取 CSV 文件...")
    articles_df = pd.read_csv('articles.csv')
    train_df = pd.read_csv('train_click_log.csv')
    test_df = pd.read_csv('testA_click_log.csv')

    click_logs_df = pd.concat([train_df, test_df], ignore_index=True)

    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(buffered=True)
        print(f"✅ 成功连接数据库，总日志数: {len(click_logs_df):,}")

        # === 1. 导入 categories ===
        print("1/8 导入 categories 表...")
        category_ids = set(safe_int(x) for x in articles_df['category_id'].dropna().unique())
        for cat_id in category_ids:
            cursor.execute("INSERT IGNORE INTO categories (category_id) VALUES (%s)", (cat_id,))
        conn.commit()

        # === 2. 导入 articles ===
        print("2/8 导入 articles 表...")
        articles_data = [
            (safe_int(row['article_id']), safe_int(row['category_id']),
             safe_int(row['created_at_ts']), safe_int(row['words_count']))
            for _, row in articles_df.iterrows()
        ]
        article_insert = """
            INSERT INTO articles (article_id, category_id, created_at_ts, words_count)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                category_id = VALUES(category_id),
                created_at_ts = VALUES(created_at_ts),
                words_count = VALUES(words_count)
        """
        cursor.executemany(article_insert, articles_data)
        conn.commit()

        # === 3. 导入 users ===
        print("3/8 导入 users 表...")
        user_ids = set(safe_int(x) for x in click_logs_df['user_id'].dropna().unique())
        cursor.executemany("INSERT IGNORE INTO users (user_id) VALUES (%s)", [(u,) for u in user_ids])
        conn.commit()

        # === 4. operating_systems (统一字符串 key) ===
        print("4/8 导入 operating_systems 表...")
        os_values = click_logs_df['click_os'].dropna().unique()
        os_id_map = {}
        for os_val in os_values:
            os_key = safe_str(os_val)
            cursor.execute("SELECT os_id FROM operating_systems WHERE os_name = %s", (os_key,))
            result = cursor.fetchone()
            if result:
                os_id_map[os_key] = result[0]
            else:
                cursor.execute("INSERT INTO operating_systems (os_name) VALUES (%s)", (os_key,))
                os_id_map[os_key] = cursor.lastrowid
        conn.commit()

        # === 5. device_groups (整数ID) ===
        print("5/8 导入 device_groups 表...")
        dg_values = set(safe_int(x) for x in click_logs_df['click_deviceGroup'].dropna().unique())
        for dg in dg_values:
            cursor.execute(
                "INSERT IGNORE INTO device_groups (device_group_id, device_group_name) VALUES (%s, %s)",
                (dg, str(dg))
            )
        conn.commit()

        # === 6. environments ===
        print("6/8 导入 environments 表...")
        env_values = set(safe_int(x) for x in click_logs_df['click_environment'].dropna().unique())
        for env in env_values:
            cursor.execute(
                "INSERT IGNORE INTO environments (env_id, env_name) VALUES (%s, %s)",
                (env, str(env))
            )
        conn.commit()

        # === 7. referrer_types ===
        print("7/8 导入 referrer_types 表...")
        ref_values = set(safe_int(x) for x in click_logs_df['click_referrer_type'].dropna().unique())
        for ref in ref_values:
            cursor.execute(
                "INSERT IGNORE INTO referrer_types (referrer_type_id, referrer_type_name) VALUES (%s, %s)",
                (ref, str(ref))
            )
        conn.commit()

        # === 8. geos (国家+地区) ===
        print("8/8 导入 geos 表...")
        geo_pairs = click_logs_df[['click_country', 'click_region']].drop_duplicates().dropna()
        geo_id_map = {}
        for _, row in geo_pairs.iterrows():
            country = safe_str(row['click_country'])
            region = safe_str(row['click_region'])
            cursor.execute(
                "SELECT geo_id FROM geos WHERE country_code = %s AND region_name = %s",
                (country, region)
            )
            result = cursor.fetchone()
            if result:
                geo_id_map[(country, region)] = result[0]
            else:
                cursor.execute(
                    "INSERT INTO geos (country_code, country_name, region_name) VALUES (%s, %s, %s)",
                    (country, country, region)
                )
                geo_id_map[(country, region)] = cursor.lastrowid
        conn.commit()

        # === 9. 分批导入 click_logs ===
        print(f"开始分批导入 click_logs 表（共 {len(click_logs_df):,} 条）...")
        insert_sql = """
            INSERT INTO click_logs (
                user_id, article_id, click_timestamp,
                env_id, device_group_id, os_id,
                geo_id, referrer_type_id
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """

        batch_size = 5000
        total = len(click_logs_df)
        processed = 0
        missing_os = set()

        for start in range(0, total, batch_size):
            batch_end = min(start + batch_size, total)
            batch_rows = click_logs_df.iloc[start:batch_end]
            batch_data = []

            for _, row in batch_rows.iterrows():
                user_id = safe_int(row['user_id'])
                article_id = safe_int(row['click_article_id'])
                ts = safe_int(row['click_timestamp'])
                env_id = safe_int(row['click_environment'])
                device_group_id = safe_int(row['click_deviceGroup'])
                referrer_id = safe_int(row['click_referrer_type'])

                # 操作系统
                os_key = safe_str(row['click_os'])
                if os_key not in os_id_map:
                    if os_key not in missing_os:
                        missing_os.add(os_key)
                        print(f"⚠️ 动态添加新操作系统: '{os_key}'")
                    cursor.execute("INSERT INTO operating_systems (os_name) VALUES (%s)", (os_key,))
                    os_id_map[os_key] = cursor.lastrowid
                os_id = os_id_map[os_key]

                # 地理位置
                country = safe_str(row['click_country'])
                region = safe_str(row['click_region'])
                geo_key = (country, region)
                if geo_key not in geo_id_map:
                    cursor.execute(
                        "INSERT INTO geos (country_code, country_name, region_name) VALUES (%s, %s, %s)",
                        (country, country, region)
                    )
                    geo_id_map[geo_key] = cursor.lastrowid
                geo_id = geo_id_map[geo_key]

                batch_data.append((
                    user_id, article_id, ts,
                    env_id, device_group_id, os_id,
                    geo_id, referrer_id
                ))

            # 插入当前批次
            cursor.executemany(insert_sql, batch_data)
            conn.commit()
            processed += len(batch_data)
            print(f"  已导入: {processed:,} / {total:,} 条 ({processed / total * 100:.1f}%)")

        print("✅ 所有数据导入成功！")

    except Error as e:
        print(f"❌ 数据库错误: {e}")
        if conn:
            try:
                conn.rollback()
            except:
                pass
    except Exception as e:
        print(f"❌ 程序错误: {repr(e)}")
        traceback.print_exc()
        if conn:
            try:
                conn.rollback()
            except:
                pass
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()
        print("数据库连接已关闭。")


if __name__ == "__main__":
    main()