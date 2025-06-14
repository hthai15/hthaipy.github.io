<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Ninja Nhảy Núi</title>
  <style>
    body {
      margin: 0;
      background: linear-gradient(to top, #0f2027, #203a43, #2c5364);
      overflow: hidden;
    }
    canvas {
      display: block;
      margin: auto;
      background-color: #87ceeb;
      border: 3px solid #222;
    }
  </style>
</head>
<body>
<canvas id="gameCanvas" width="400" height="600"></canvas>
<script>
const canvas = document.getElementById("gameCanvas");
const ctx = canvas.getContext("2d");

let ninja = {
  x: 180,
  y: 500,
  width: 30,
  height: 30,
  color: "#000",
  velocityY: 0,
  jumpPower: -10,
  gravity: 0.5
};

let platforms = [];
let score = 0;
let isGameOver = false;

function createPlatform(y) {
  const width = Math.random() * 60 + 60;
  const x = Math.random() * (canvas.width - width);
  platforms.push({ x, y, width, height: 10 });
}

function initPlatforms() {
  for (let i = 0; i < 6; i++) {
    createPlatform(600 - i * 100);
  }
}

function drawNinja() {
  ctx.fillStyle = ninja.color;
  ctx.fillRect(ninja.x, ninja.y, ninja.width, ninja.height);
}

function drawPlatforms() {
  ctx.fillStyle = "#444";
  platforms.forEach(p => {
    ctx.fillRect(p.x, p.y, p.width, p.height);
  });
}

function updatePlatforms() {
  platforms.forEach(p => {
    p.y += 2; // move down to simulate ninja moving up
  });
  platforms = platforms.filter(p => p.y < canvas.height);

  if (platforms.length < 6) {
    createPlatform(-10);
  }
}

function checkCollision() {
  if (ninja.velocityY > 0) {
    platforms.forEach(p => {
      if (
        ninja.x + ninja.width > p.x &&
        ninja.x < p.x + p.width &&
        ninja.y + ninja.height >= p.y &&
        ninja.y + ninja.height <= p.y + p.height
      ) {
        ninja.velocityY = ninja.jumpPower;
        score++;
      }
    });
  }
}

function updateNinja() {
  ninja.velocityY += ninja.gravity;
  ninja.y += ninja.velocityY;

  if (ninja.y > canvas.height) {
    isGameOver = true;
  }
}

function drawScore() {
  ctx.fillStyle = "#fff";
  ctx.font = "20px Arial";
  ctx.fillText("✨ Điểm: " + score, 10, 30);
}

function drawGameOver() {
  ctx.fillStyle = "#fff";
  ctx.font = "40px Arial";
  ctx.fillText("Game Over", 100, 300);
  ctx.font = "20px Arial";
  ctx.fillText("Nhấn Space để chơi lại", 110, 340);
}

function gameLoop() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!isGameOver) {
    updateNinja();
    updatePlatforms();
    checkCollision();
    drawPlatforms();
    drawNinja();
    drawScore();
    requestAnimationFrame(gameLoop);
  } else {
    drawPlatforms();
    drawNinja();
    drawScore();
    drawGameOver();
  }
}

document.addEventListener("keydown", e => {
  if (e.code === "Space") {
    if (isGameOver) {
      ninja.y = 500;
      ninja.velocityY = 0;
      score = 0;
      platforms = [];
      initPlatforms();
      isGameOver = false;
      gameLoop();
    } else if (ninja.velocityY > 0) {
      ninja.velocityY = ninja.jumpPower;
    }
  }
});

initPlatforms();
gameLoop();
</script>
</body>
</html>
