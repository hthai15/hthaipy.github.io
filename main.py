import streamlit as st
import random
from datetime import datetime
import pandas as pd
import os

START_MONEY = 1000
RANKING_FILE = "ranking.csv"
MAX_BULLETS = 3

def initialize_session():
    if "users" not in st.session_state:
        st.session_state.users = {}
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "money" not in st.session_state:
        st.session_state.money = START_MONEY
    if "shots_left" not in st.session_state:
        st.session_state.shots_left = MAX_BULLETS
    if "target_value" not in st.session_state:
        st.session_state.target_value = random.randint(1, 10)
    if "game_log" not in st.session_state:
        st.session_state.game_log = []

def load_ranking():
    if not os.path.exists(RANKING_FILE):
        df = pd.DataFrame(columns=["username", "best_money"])
        df.to_csv(RANKING_FILE, index=False)
    return pd.read_csv(RANKING_FILE)

def save_ranking(username, money):
    df = load_ranking()
    if username in df["username"].values:
        idx = df.index[df["username"] == username][0]
        if money > df.at[idx, "best_money"]:
            df.at[idx, "best_money"] = money
    else:
        df.loc[len(df.index)] = [username, money]
    df.to_csv(RANKING_FILE, index=False)

def login_page():
    st.title("🔫 Trò chơi bắn súng - Đăng nhập/Đăng ký")
    username = st.text_input("Tên đăng nhập")
    password = st.text_input("Mật khẩu", type="password")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Đăng nhập"):
            if username in st.session_state.users and st.session_state.users[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.money = START_MONEY
                st.session_state.shots_left = MAX_BULLETS
                st.session_state.target_value = random.randint(1, 10)
                st.success(f"Đăng nhập thành công! Chào {username}")
                st.experimental_rerun()
            else:
                st.error("Sai tên đăng nhập hoặc mật khẩu.")

    with col2:
        if st.button("Đăng ký"):
            if username == "" or password == "":
                st.warning("Vui lòng nhập đủ tên đăng nhập và mật khẩu.")
            elif username in st.session_state.users:
                st.error("Tên đăng nhập đã tồn tại.")
            else:
                st.session_state.users[username] = password
                st.success("Đăng ký thành công! Vui lòng đăng nhập.")
                st.experimental_rerun()

def game_page():
    st.title(f"🎯 Trò chơi bắn súng - Chào {st.session_state.username}")
    st.write(f"💰 Số xu hiện tại: **{st.session_state.money} xu**")
    st.write(f"🔫 Đạn còn lại: **{st.session_state.shots_left}/{MAX_BULLETS}**")

    if st.session_state.shots_left == 0:
        st.warning("🎮 Hết lượt bắn. Nhấn nút để chơi lại.")
        if st.button("🔁 Bắt đầu lại"):
            st.session_state.shots_left = MAX_BULLETS
            st.session_state.target_value = random.randint(1, 10)
            st.session_state.game_log = []
            st.experimental_rerun()
        show_log()
        show_ranking()
        return

    st.markdown("---")
    st.subheader("🎯 Nhắm bắn vào mục tiêu")
    st.write("Chọn lực bắn từ 1 đến 10. Nếu lực bắn đúng bằng giá trị mục tiêu, bạn trúng!")
    force = st.slider("Lực bắn", 1, 10, 5)

    if st.button("🔥 Bắn!"):
        hit = force == st.session_state.target_value
        log_entry = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "force": force,
            "target": st.session_state.target_value,
            "result": "🎯 Trúng!" if hit else "💨 Trượt!"
        }
        st.session_state.game_log.append(log_entry)
        if hit:
            st.success("🎯 Bắn trúng! +100 xu")
            st.session_state.money += 100
        else:
            st.error("💨 Bắn trượt! Không được xu")
        st.session_state.shots_left -= 1
        st.session_state.target_value = random.randint(1, 10)
        save_ranking(st.session_state.username, st.session_state.money)
        st.experimental_rerun()

    show_log()
    show_ranking()

def show_log():
    st.subheader("📜 Lịch sử bắn")
    if not st.session_state.game_log:
        st.write("Chưa có lượt bắn nào.")
        return
    for log in reversed(st.session_state.game_log):
        st.write(f"{log['time']} | Lực bắn: {log['force']} | Mục tiêu: {log['target']} | Kết quả: {log['result']}")

def show_ranking():
    st.subheader("🏆 Bảng xếp hạng")
    df = load_ranking()
    if df.empty:
        st.write("Chưa có dữ liệu bảng xếp hạng.")
        return
    df = df.sort_values(by="best_money", ascending=False).head(5)
    st.table(df)

def main():
    st.set_page_config(page_title="Trò chơi bắn súng", layout="centered")
    initialize_session()
    if not st.session_state.logged_in:
        login_page()
    else:
        game_page()

if __name__ == "__main__":
    main()
