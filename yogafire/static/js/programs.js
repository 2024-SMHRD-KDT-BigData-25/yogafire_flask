// 특정 조건이 충족되었을 때 실행되는 함수
function addFireEffect() {
    // fire 요소 생성
    const fireElement = document.createElement('div');
    fireElement.className = 'fire';
    fireElement.innerHTML = `
        <div class="fire-left">
            <div class="main-fire"></div>
            <div class="particle-fire"></div>
        </div>
        <div class="fire-center">
            <div class="main-fire"></div>
            <div class="particle-fire"></div>
        </div>
        <div class="fire-right">
            <div class="main-fire"></div>
            <div class="particle-fire"></div>
        </div>
        <div class="fire-bottom">
            <div class="main-fire"></div>
        </div>
    `;

    // 이미지들 사이에 삽입
    const yogaValueDiv = document.querySelector('.yogavalue');
    const images = yogaValueDiv.querySelectorAll('.yogaimg');
    
    // 원하는 위치에 삽입 (예: 두 번째 이미지 뒤)
    if (images.length >= 2) {
        images[1].after(fireElement); // 두 번째 이미지 뒤에 삽입
    } else {
        yogaValueDiv.appendChild(fireElement); // 이미지가 적으면 맨 마지막에 삽입
    }
}

// 이벤트 예제: 버튼 클릭 시 fire 효과 추가
document.querySelector('.button-group button').addEventListener('click', () => {
    addFireEffect();
});
