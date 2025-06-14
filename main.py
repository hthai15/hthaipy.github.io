<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8" />
  <title>🎮 Game Tâng Bóng Qua Chướng Ngại Vật</title>
  <style>
    body {
      margin: 0;
      overflow: hidden;
      background: linear-gradient(to bottom, #87ceeb, #ffffff);
      font-family: 'Segoe UI', sans-serif;
    }
    canvas {
      display: block;
      margin: auto;
      background: #dff9fb;
      border: 3px solid #0984e3;
      border-radius: 10px;
    }
    #score {
      position: absolute;
      top: 10px;
      left: 10px;
      font-size: 24px;
      color: #2d3436;
    }
    #game-over {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      font-size: 32px;
      color: red;
      display: none;
    }
  </style>
</head>
<body>
  <div id="score">Điểm: 0</div>
  <div id="game-over">🎮 Game Over! Nhấn Space để chơi lại.</div>
  <canvas id="gameCanvas" width="480" height="640"></canvas>

  <script>
    const canvas = document.getElementById("gameCanvas");
    const ctx = canvas.getContext("2d");

    const GRAVITY = 0.15;
    const JUMP = -6.5;
    const MAX_FALL_SPEED = 4;
    const OBSTACLE_WIDTH = 60;
    const OBSTACLE_GAP = 160;
    const OBSTACLE_SPEED = 2;

    let ball = {
      x: 80,
      y: canvas.height / 2,
      radius: 15,
      velocity: 0,
    };

    let obstacles = [];
    let score = 0;
    let gameOver = false;

    function resetGame() {
      ball.y = canvas.height / 2;
      ball.velocity = 0;
      obstacles = [];
      score = 0;
      gameOver = false;
      document.getElementById("game-over").style.display = "none";
    }

    function createObstacle() {
      const topHeight = Math.random() * (canvas.height - OBSTACLE_GAP - 100) + 50;
      obstacles.push({
        x: canvas.width,
        topHeight: topHeight,
        bottomY: topHeight + OBSTACLE_GAP,
      });
    }

    function update() {
      if (gameOver) return;

      ball.velocity += GRAVITY;
      if (ball.velocity > MAX_FALL_SPEED) {
        ball.velocity = MAX_FALL_SPEED;
      }
      ball.y += ball.velocity;

      // Tạo chướng ngại vật mới
      if (obstacles.length === 0 || obstacles[obstacles.length - 1].x < canvas.width - 200) {
        createObstacle();
      }

      // Cập nhật chướng ngại vật
      for (let i = 0; i < obstacles.length; i++) {
        obstacles[i].x -= OBSTACLE_SPEED;

        // Kiểm tra vượt qua chướng ngại
        if (!obstacles[i].passed && obstacles[i].x + OBSTACLE_WIDTH < ball.x) {
          obstacles[i].passed = true;
          score++;
          document.getElementById("score").textContent = "Điểm: " + score;
        }

        // Kiểm tra va chạm
        if (
          ball.x + ball.radius > obstacles[i].x &&
          ball.x - ball.radius < obstacles[i].x + OBSTACLE_WIDTH
        ) {
          if (
            ball.y - ball.radius < obstacles[i].topHeight ||
            ball.y + ball.radius > obstacles[i].bottomY
          ) {
            endGame();
          }
        }
      }

      // Xóa vật cản ngoài màn hình
      obstacles = obstacles.filter((obs) => obs.x + OBSTACLE_WIDTH > 0);

      // Va chạm sàn hoặc trần
      if (ball.y + ball.radius >= canvas.height || ball.y - ball.radius <= 0) {
        endGame();
      }
    }

    function draw() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Vẽ bóng
      ctx.beginPath();
      ctx.arc(ball.x, ball.y, ball.radius, 0, Math.PI * 2);
      ctx.fillStyle = "#e17055";
      ctx.fill();
      ctx.closePath();

      // Vẽ chướng ngại vật
      for (let obs of obstacles) {
        ctx.fillStyle = "#2d3436";
        ctx.fillRect(obs.x, 0, OBSTACLE
