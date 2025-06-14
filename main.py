import streamlit as st
import random
import streamlit.components.v1 as components

st.set_page_config(layout="wide")
st.title("🔫 Game Điều Khiển Súng Bắn Mục Tiêu")

# Tọa độ mục tiêu (ngẫu nhiên mỗi lần load)
if "target_x" not in st.session_state or "target_y" not in st.session_state:
    st.session_state.target_x = random.randint(50, 500)
    st.session_state.target_y = random.randint(50, 300)

# Điểm số
if "score" not in st.session_state:
    st.session_state.score = 0

# Vẽ giao diện HTML Canvas + JavaScript
canvas_html = f"""
<canvas id="shootCanvas" width="600" height="400" style="border:1px solid #000;"></canvas>
<p id="status">Bấm chuột để bắn!</p>

<script>
let canvas = document.getElementById("shootCanvas");
let ctx = canvas.getContext("2d");

// Vẽ mục tiêu
function drawTarget(x, y) {{
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    ctx.arc(x, y, 20, 0, 2 * Math.PI);
    ctx.fillStyle = "red";
    ctx.fill();
}}

drawTarget({st.session_state.target_x}, {st.session_state.target_y});

// Sự kiện bắn
canvas.addEventListener("click", function(event) {{
    const rect = canvas.getBoundingClientRect();
    const clickX = event.clientX - rect.left;
    const clickY = event.clientY - rect.top;

    // Gửi tọa độ về Streamlit
    const data = {{
        x: clickX,
        y: clickY
    }};
    fetch("/", {{
        method: "POST",
        headers: {{
            "Content-Type": "application/json"
        }},
        body: JSON.stringify({{eventType: "click", data: data}})
    }});

    window.location.reload();
}});
</script>
"""

# Hiển thị canvas
components.html(canvas_html, height=450)

# Lấy dữ liệu tọa độ click
if "x" in st.experimental_get_query_params():
    click_x = int(st.experimental_get_query_params()["x"][0])
    click_y = int(st.experimental_get_query_params()["y"][0])

    # So sánh khoảng cách đến mục tiêu
    dx = click_x - st.session_state.target_x
    dy = click_y - st.session_state.target_y
    distance = (dx ** 2 + dy ** 2) ** 0.5

    if distance < 30:
        st.success("🎯 Trúng mục tiêu! +1 điểm")
        st.session_state.score += 1
    else:
        st.error("💨 Trượt mục tiêu!")

    # Tạo mục tiêu mới
    st.session_state.target_x = random.randint(50, 500)
    st.session_state.target_y = random.randint(50, 300)

# Hiển thị điểm
st.markdown(f"## ⭐ Điểm hiện tại: {st.session_state.score}")
