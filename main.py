import streamlit as st

# Cấu hình trang
st.set_page_config(page_title="Bán Bot Lập Trình", layout="wide")

# Tiêu đề trang
st.markdown("<h1 style='text-align:center; color:#4A90E2;'>🤖 Bán Bot Lập Trình Tự Động</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Chúng tôi cung cấp các bot thông minh cho lập trình, tự động hóa và xử lý công việc!</p>", unsafe_allow_html=True)
st.markdown("---")

# Giới thiệu sản phẩm
st.subheader("🛍️ Sản phẩm nổi bật")
cols = st.columns(3)

with cols[0]:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712027.png", width=100)
    st.markdown("### Bot ChatGPT API")
    st.write("Trả lời tự động, xử lý ngôn ngữ tự nhiên, tích hợp website, Facebook.")
    
with cols[1]:
    st.image("https://cdn-icons-png.flaticon.com/512/5587/5587464.png", width=100)
    st.markdown("### Bot Telegram Tự Động")
    st.write("Gửi tin nhắn, trả lời lệnh, truy xuất dữ liệu, quản lý nhóm.")

with cols[2]:
    st.image("https://cdn-icons-png.flaticon.com/512/2311/2311524.png", width=100)
    st.markdown("### Bot Discord Developer")
    st.write("Auto mod, game mini, phản hồi theo lệnh, tích hợp AI/ML.")

st.markdown("---")

# Bảng giá
st.subheader("💰 Bảng Giá Gói Dịch Vụ")

price_cols = st.columns(3)

with price_cols[0]:
    st.markdown("### 🟢 Basic")
    st.write("- 1 nền tảng (Telegram hoặc Discord)")
    st.write("- Xử lý lệnh cơ bản")
    st.write("- Hỗ trợ 24/7")
    st.success("Giá: 499.000 VNĐ")

with price_cols[1]:
    st.markdown("### 🟡 Pro")
    st.write("- 2 nền tảng")
    st.write("- ChatGPT tích hợp")
    st.write("- Lưu dữ liệu & báo cáo")
    st.success("Giá: 999.000 VNĐ")

with price_cols[2]:
    st.markdown("### 🔴 Enterprise")
    st.write("- Tuỳ chỉnh theo yêu cầu")
    st.write("- Hệ thống back-end riêng")
    st.write("- Hỗ trợ mở rộng")
    st.success("Giá: Liên hệ : 0816157172")

st.markdown("---")

# Form liên hệ
st.subheader("📬 Liên hệ đặt hàng hoặc yêu cầu riêng")

with st.form("contact_form"):
    name = st.text_input("Họ tên")
    email = st.text_input("Email")
    need = st.text_area("Yêu cầu của bạn")

    submitted = st.form_submit_button("Gửi yêu cầu")
    if submitted:
        if name and email and need:
            st.success("✅ Yêu cầu đã được gửi! Chúng tôi sẽ liên hệ bạn sớm.")
        else:
            st.error("❗ Vui lòng điền đầy đủ thông tin.")

st.markdown("---")
st.markdown("<p style='text-align:center'>© 2025 Bán Bot Lập Trình. All rights reserved.</p>", unsafe_allow_html=True)
