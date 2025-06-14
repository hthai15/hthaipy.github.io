import streamlit as st
import random
from datetime import datetime
import pandas as pd
import os

START_MONEY = 1000
RANKING_FILE = "ranking.csv"

DICE_IMAGES = {
    1: "https://upload.wikimedia.org/wikipedia/commons/1/1b/Dice-1-b.svg",
    2: "https://upload.wikimedia.org/wikipedia/commons/5/5f/Dice-2-b.svg",
    3: "https://upload.wikimedia.org/wikipedia/commons/b/b1/Dice-3-b.svg",
    4: "https://upload.wikimedia.org/wikipedia/commons/f/fd/Dice-4-b.svg",
    5: "https://upload.wikimedia.org/wikipedia/commons/0/08/Dice-5-b.svg",
    6: "https://upload.wikimedia.org/wikipedia/commons/2/26/Dice-6-b.svg",
}

def initialize_session():
    if "users" not in st.session_state:
        st.session_state.users = {}
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "money" not in st.session_state:
        st.session_state.money = 0
    if "history" not in st.session_state:
        st.session_state.history = []
    if "round_start_time" not in st.session_state:
        st.session_state.round_start_time = None
    if "bet" not in st.session_state:
        st.session_state.bet = 0
    if "choice" not in st.session_state:
        st.session_state.choice = "Tài (11-18)"
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "last_dice" not in st.session_state:
        st.session_state.last_dice = []
    if "last_total" not in st.session_state:
        st.session_state.last_total = 0
    if "last_win" not in st.session_state:
        st.session_state.last_win = 0

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
        df = df.append({"username": username, "best_money": money}, ignore_index=True)
    df.to_csv(RANKING_FILE, index=False)

def add_history(bet, choice, dice, total, result, win_amount):
    st.session_state.history.append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "bet": bet,
        "choice": choice,
        "dice": dice,
        "total": total,
        "result": result,
        "win_amount": win_amount,
    })

def show_history():
    st.subheader("📜 Lịch sử chơi")
    if len(st.session_state.history) == 0:
        st.write("Chưa có lượt chơi nào.")
        return
    for record in reversed(st.session_state.history):
        st.write(f"{record['time']} | Cược: {record['bet']} | Chọn: {record['choice']} | "
                 f"Xúc xắc: {record['dice']} (Tổng: {record['total']}) | Kết quả: {record['result']} | "
                 f"Thắng/Lỗ: {record['win_amount']} xu")

def show_ranking():
    st.subheader("🏆 Bảng xếp hạng")
    df = load_ranking()
    if df.empty:
        st.write("Chưa có dữ liệu bảng xếp hạng.")
        return
    df = df.sort_values(by="best_money", ascending=False).head(5)
    st.table(df)

def login_page():
    st.title("🎲 Game Tài Xỉu - Đăng nhập/Đăng ký")
    st.write("Vui lòng đăng nhập hoặc đăng ký để chơi.")

    username = st.text_input("Tên đăng nhập")
    password = st.text_input("Mật khẩu", type="password")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Đăng nhập"):
            if username in st.session_state.users and st.session_state.users[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.money = START_MONEY
                st.session_state.history = []
                st.session_state.round_start_time = datetime.now()
                st.session_state.bet = 0
                st.session_state.choice = "Tài (11-18)"
                st.session_state.last_result = None
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
    st.title(f"🎲 Game Tài Xỉu - Chào bạn {st.session_state.username}")
    st.write(f"💰 Số dư hiện tại: **{st.session_state.money} xu**")

    # Khởi tạo thời gian bắt đầu vòng nếu chưa có
    if st.session_state.round_start_time is None:
        st.session_state.round_start_time = datetime.now()

    now = datetime.now()
    elapsed = (now - st.session_state.round_start_time).total_seconds()
    remaining = 60 - elapsed

    # Khi hết 60s thì tung xúc xắc và bắt đầu vòng mới
    if remaining <= 0:
        dice = [random.randint(1, 6) for _ in range(3)]
        total = sum(dice)
        result = "Tài" if total >= 11 else "Xỉu"

        bet = st.session_state.bet
        choice = st.session_state.choice
        if bet > 0:
            if choice.startswith(result):
                win_amount = bet
                st.session_state.money += win_amount
                st.success(f"🎉 Vòng trước bạn thắng {win_amount} xu!")
            else:
                win_amount = -bet
                st.session_state.money += win_amount
                st.warning(f"😢 Vòng trước bạn thua {bet} xu!")
            add_history(bet, choice, dice, total, result, win_amount)
        else:
            win_amount = 0
            st.info("Vòng trước bạn không đặt cược.")

        st.session_state.last_dice = dice
        st.session_state.last_total = total
        st.session_state.last_result = result
        st.session_state.last_win = win_amount

        st.session_state.bet = 0
        st.session_state.choice = "Tài (11-18)"
        st.session_state.round_start_time = datetime.now()
        remaining = 60

        st.experimental_rerun()

    st.markdown(f"⏳ Thời gian còn lại: **{int(remaining)} giây**")

    betting_allowed = remaining > 5

    with st.form(key="bet_form"):
        if betting_allowed:
            bet = st.number_input(
                "Nhập số xu cược",
                min_value=1,
                max_value=st.session_state.money,
                value=st.session_state.bet if st.session_state.bet > 0 else 1,
                step=1,
                key="bet_input"
            )
            choice = st.selectbox(
                "Chọn Tài hoặc Xỉu",
                ("Tài (11-18)", "Xỉu (3-10)"),
                index=0 if st.session_state.choice.startswith("Tài") else 1,
                key="choice_input"
            )
            submitted = st.form_submit_button("Đặt cược")
            if submitted:
                if bet > st.session_state.money:
                    st.error("Bạn không đủ xu để cược số tiền này.")
                else:
                    st.session_state.bet = bet
                    st.session_state.choice = choice
                    st.success(f"Bạn đã đặt cược {bet} xu cho {choice}.")
        else:
            st.info("⏰ Đã ngừng nhận cược vì chỉ còn dưới 5 giây!")

    # Hiển thị kết quả vòng trước (nếu có)
    if st.session_state.last_result:
        st.markdown("---")
        st.subheader("Kết quả vòng trước:")
        dice_imgs = ""
        for d in st.session_state.last_dice:
            dice_imgs += f'<img src="{DICE_IMAGES[d]}" width="50" style="margin-right:10px"/>'
        st.markdown(dice_imgs, unsafe_allow_html=True)
        st.write(f"Tổng điểm: **{st.session_state.last_total}**")
        st.write(f"Kết quả: **{st.session_state.last_result}**")
        if st.session_state.last_win >= 0:
            st.success(f"Bạn thắng: {st.session_state.last_win} xu")
        else:
            st.error(f"Bạn thua: {-st.session_state.last_win} xu")

    st.markdown("---")
    show_history()
    st.markdown("---")
    show_ranking()

def main():
    st.set_page_config(page_title="Game Tài Xỉu", layout="centered")
    initialize_session()

    if not st.session_state.logged_in:
        login_page()
    else:
        game_page()

if __name__ == "__main__":
    main()
