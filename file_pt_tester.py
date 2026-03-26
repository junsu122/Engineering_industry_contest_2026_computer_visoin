

import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO

# 1. 모델 로드 (본인의 best.pt 경로로 수정)
model = YOLO('/home/junsu/engineering_contest/Engineering_industry_contest_2026_computer_visoin/best(1).pt')

# 2. 화면 캡처 설정
# {'top': 시작y, 'left': 시작x, 'width': 가로, 'height': 세로}
monitor = {"top": 0, "left": 0, "width": 960, "height": 1080}

with mss() as sct:
    while True:
        # 화면 캡처
        img = np.array(sct.grab(monitor))

        # MSS는 BGRA 형태로 캡처하므로 BGR로 변환 (YOLO 입력용)
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # 3. YOLO 추론 수행
        # stream=True는 메모리 효율을 위해 권장됩니다.
        results = model.predict(frame, conf=0.5, show=False)

        # 결과 렌더링 (Annotated frame)
        annotated_frame = results[0].plot()

        # 4. 화면 출력
        cv2.imshow("Screen Object Detection", annotated_frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()