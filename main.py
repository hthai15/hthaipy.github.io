import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Ninja Nhảy Núi", layout="centered")

st.title("🥷 Ninja Nhảy Núi")
st.markdown("**Hướng dẫn:** Nhấn **Space** hoặc **Click chuột** để nhảy. Đừng rơi khỏi màn hình nhé!")

game_html = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Ninja Nhảy Núi</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    canvas {
      display: block;
      margin: auto;
      background: linear-gradient(to top, #2b1055, #7597de);
      border: 4px solid #333;
    }
    body {
      background: #111;
      color: white;
      font-family: Arial, sans-serif;
      text-align: center;
    }
    #restartBtn {
      display: none;
      margin-top: 20px;
      padding: 10px 20px;
      font-size: 18px;
      background-color: #4CAF50;
      color: white;
      border: none;
      border-radius: 8px;
      cursor: pointer;
    }
    #restartBtn:hover {
      background-color: #45a049;
    }
  </style>
</head>
<body>
  <canvas id="gameCanvas" width="400" height="600"></canvas>
  <button id="restartBtn" onclick="restartGame()">🔁 Chơi lại</button>

  <script>
    const canvas = document.getElementById("gameCanvas");
    const ctx = canvas.getContext("2d");
    const restartBtn = document.getElementById("restartBtn");

    let ninja;
    let platforms = [];
    let score;
    let gameOver;

    function initGame() {
      ninja = {
        x: 200,
        y: 500,
        width: 30,
        height: 30,
        color: "#f44336",
        velocityY: 0,
        jumpPower: -10,
        gravity: 0.5
      };

      platforms = [];
      score = 0;
      gameOver = false;

      for (let i = 0; i < 6; i++) {
        platforms.push({
          x: Math.random() * 300,
          y: 600 - i * 100,
          width: 100,
          height: 10
        });
      }

      restartBtn.style.display = "none";
      update();
    }

    function drawNinja() {
      ctx.fillStyle = ninja.color;
      ctx.fillRect(ninja.x, ninja.y, ninja.width, ninja.height);
    }

    function drawPlatforms() {
      ctx.fillStyle = "#4caf50";
      platforms.forEach(p => {
        ctx.fillRect(p.x, p.y, p.width, p.height);
      });
    }

    function updatePlatforms() {
      platforms.forEach(p => {
        p.y += 2;
        if (p.y > canvas.height) {
          p.y = 0;
          p.x = Math.random() * 300;
          score++;
        }
      });
    }

    function checkCollision() {
      platforms.forEach(p => {
        if (
          ninja.y + ninja.height < p.y + 5 &&
          ninja.y + ninja.height + ninja.velocityY >= p.y &&
          ninja.x + ninja.width > p.x &&
          ninja.x < p.x + p.width
        ) {
          ninja.velocityY = ninja.jumpPower;
        }
      });
    }

    function update() {
      if (gameOver) return;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      ninja.velocityY += ninja.gravity;
      ninja.y += ninja.velocityY;

      updatePlatforms();
      checkCollision();
      drawPlatforms();
      drawNinja();

      ctx.fillStyle = "white";
      ctx.font = "20px Arial";
      ctx.fillText("Điểm: " + score, 10, 30);

      if (ninja.y > canvas.height) {
        gameOver = true;
        ctx.fillStyle = "#ff3333";
        ctx.font = "36px Arial";
        ctx.fillText("Game Over", 100, 300);
        restartBtn.style.display = "inline-block";
        return;
      }

      requestAnimationFrame(update);
    }

    function jump() {
      if (!gameOver) {
        ninja.velocityY = ninja.jumpPower;
      }
    }

    function restartGame() {
      initGame();
    }

    document.addEventListener("keydown", (e) => {
      if (e.code === "Space") jump();
    });
    canvas.addEventListener("mousedown", jump);

    initGame();
  </script>
</body>
</html>
"""

components.html(game_html, height=700)
