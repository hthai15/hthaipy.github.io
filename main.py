import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Game Tưng Bóng", layout="centered")

st.title("🏀 Trò chơi Tưng Bóng")
st.markdown("**Hướng dẫn:** Dùng các phím ← và → để điều khiển bóng. Đừng để bóng rơi ra ngoài!")

game_html = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    canvas {
      background: #eef;
      display: block;
      margin: 0 auto;
      border: 2px solid #444;
    }
    body {
      text-align: center;
      font-family: sans-serif;
    }
  </style>
</head>
<body>
<canvas id="gameCanvas" width="400" height="500"></canvas>
<script>
  const canvas = document.getElementById("gameCanvas");
  const ctx = canvas.getContext("2d");

  let ball = {
    x: 200,
    y: 100,
    radius: 20,
    vx: 2,
    vy: 0,
    gravity: 0.5,
    bounce: -0.7
  };

  let score = 0;

  function drawBall() {
    ctx.beginPath();
    ctx.arc(ball.x, ball.y, ball.radius, 0, Math.PI * 2);
    ctx.fillStyle = "#FF5722";
    ctx.fill();
    ctx.closePath();
  }

  function drawScore() {
    ctx.font = "18px Arial";
    ctx.fillStyle = "#333";
    ctx.fillText("Điểm: " + score, 10, 20);
  }

  function update() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ball.vy += ball.gravity;
    ball.y += ball.vy;
    ball.x += ball.vx;

    // Tường trái/phải
    if (ball.x + ball.radius > canvas.width || ball.x - ball.radius < 0) {
      ball.vx = -ball.vx;
    }

    // Nền
    if (ball.y + ball.radius > canvas.height) {
      ball.y = canvas.height - ball.radius;
      ball.vy *= ball.bounce;
      score += 1;
    }

    drawBall();
    drawScore();
    requestAnimationFrame(update);
  }

  // Điều khiển phím
  document.addEventListener("keydown", function (e) {
    if (e.key === "ArrowLeft") {
      ball.vx -= 1;
    }
    if (e.key === "ArrowRight") {
      ball.vx += 1;
    }
  });

  update();
</script>
</body>
</html>
"""

# Nhúng HTML game
components.html(game_html, height=550)
