import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Tâng Bóng Vượt Chướng Ngại Vật", layout="centered")

st.title("🏐 Tâng Bóng Qua Chướng Ngại Vật")
st.markdown("**Hướng dẫn:** Bấm Space hoặc nhấn chuột để tâng bóng. Đừng để bóng rơi hoặc va vào chướng ngại vật!")

# HTML/JS game embedded
game_html = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    canvas {
      background: linear-gradient(to bottom, #b3ecff, #e6faff);
      display: block;
      margin: auto;
      border: 2px solid #333;
    }
    body {
      margin: 0;
      text-align: center;
      font-family: Arial;
    }
  </style>
</head>
<body>
<canvas id="gameCanvas" width="400" height="600"></canvas>
<script>
  const canvas = document.getElementById("gameCanvas");
  const ctx = canvas.getContext("2d");

  const GRAVITY = 0.35; // giảm tốc độ rơi
  const JUMP = -7;
  let score = 0;
  let gameOver = false;

  const ball = {
    x: 80,
    y: 300,
    radius: 15,
    velocity: 0
  };

  const pipes = [];
  const pipeWidth = 60;
  const gap = 160;
  let frame = 0;

  function drawBall() {
    ctx.beginPath();
    ctx.arc(ball.x, ball.y, ball.radius, 0, Math.PI * 2);
    ctx.fillStyle = "#ff5722";
    ctx.fill();
    ctx.closePath();
  }

  function drawPipes() {
    pipes.forEach(pipe => {
      ctx.fillStyle = "#4CAF50";
      ctx.fillRect(pipe.x, 0, pipeWidth, pipe.top);
      ctx.fillRect(pipe.x, pipe.top + gap, pipeWidth, canvas.height);
    });
  }

  function drawScore() {
    ctx.font = "20px Arial";
    ctx.fillStyle = "#333";
    ctx.fillText("Điểm: " + score, 10, 30);
  }

  function drawGameOver() {
    ctx.font = "40px Arial";
    ctx.fillStyle = "#ff3333";
    ctx.fillText("Game Over", 90, 300);
    ctx.font = "20px Arial";
    ctx.fillStyle = "#555";
    ctx.fillText("Bấm Space hoặc Click để chơi lại", 60, 340);
  }

  function update() {
    if (gameOver) return;

    ball.velocity += GRAVITY;
    ball.y += ball.velocity;

    // Tạo ống mới
    if (frame % 100 === 0) {
      const topHeight = Math.floor(Math.random() * 250) + 50;
      pipes.push({ x: canvas.width, top: topHeight, passed: false });
    }

    // Cập nhật ống
    pipes.forEach(pipe => {
      pipe.x -= 2;

      // Kiểm tra va chạm
      if (
        ball.x + ball.radius > pipe.x && ball.x - ball.radius < pipe.x + pipeWidth &&
        (ball.y - ball.radius < pipe.top || ball.y + ball.radius > pipe.top + gap)
      ) {
        gameOver = true;
      }

      // Tính điểm
      if (!pipe.passed && pipe.x + pipeWidth < ball.x) {
        score++;
        pipe.passed = true;
      }
    });

    // Va chạm với tường trên/dưới
    if (ball.y + ball.radius > canvas.height || ball.y - ball.radius < 0) {
      gameOver = true;
    }

    // Xóa ống đã đi qua
    if (pipes.length > 0 && pipes[0].x + pipeWidth < 0) {
      pipes.shift();
    }

    draw();
    frame++;
    requestAnimationFrame(update);
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawBall();
    drawPipes();
    drawScore();

    if (gameOver) {
      drawGameOver();
    }
  }

  function jump() {
    if (!gameOver) {
      ball.velocity = JUMP;
    } else {
      // Reset game
      ball.y = 300;
      ball.velocity = 0;
      pipes.length = 0;
      score = 0;
      frame = 0;
      gameOver = false;
      update();
    }
  }

  document.addEventListener("keydown", (e) => {
    if (e.code === "Space") jump();
  });

  canvas.addEventListener("mousedown", jump);

  update();
</script>
</body>
</html>
"""

components.html(game_html, height=650)
