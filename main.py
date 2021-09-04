import cv2
import numpy as np
import utils


def scan_document():
    ########################################################################
    cap = cv2.VideoCapture("outpy.avi")
    cap.set(10, 160)
    frame_height = 640
    frame_width = 480
    img_warp_colored = None
    ########################################################################

    utils.initialize_trackbars()
    count = 0

    while True:
        success, img = cap.read()
        img = cv2.resize(img, (frame_width, frame_height))
        blank_image = np.zeros((frame_height, frame_width, 3),
                               np.uint8)  # 書類を見つからない場合黒の画像を表示
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 二値化
        blur_image = cv2.GaussianBlur(gray_image, (5, 5), 1)
        edge_threshold = utils.val_trackbars()  # エッジの閾値
        img_threshold = cv2.Canny(blur_image, edge_threshold[0], edge_threshold[1])
        kernel = np.ones((5, 5))

        # クロージング
        dilated_image = cv2.dilate(img_threshold, kernel, iterations=2)  # 膨張
        img_threshold = cv2.erode(dilated_image, kernel, iterations=2)  # 収縮

        contour_image = img.copy()  # 輪郭表示用
        img_big_contour = img.copy()
        contours, hierarchy = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[
                              -2:]
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 10)  # 全ての輪郭を描画

        # 一番大きい輪郭を抽出
        biggest, max_area = utils.biggest_contour(contours)
        if biggest.size != 0:
            biggest = utils.reorder(biggest)
            cv2.drawContours(img_big_contour, biggest, -1, (0, 255, 0), 20)
            img_big_contour = utils.draw_rectangle(img_big_contour, biggest, 2)

            pts1 = np.float32(biggest)
            pts2 = np.float32(
                [[0, 0], [frame_width, 0], [0, frame_height], [frame_width, frame_height]])

            # 歪みを修正
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            img_warp_colored = cv2.warpPerspective(img, matrix, (frame_width, frame_height))

            img_warp_colored = img_warp_colored[20:img_warp_colored.shape[0] - 20, 20:img_warp_colored.shape[1] - 20]
            img_warp_colored = cv2.resize(img_warp_colored, (frame_width, frame_height))

            img_warp_gray = cv2.cvtColor(img_warp_colored, cv2.COLOR_BGR2GRAY)
            img_adaptive_threshold = cv2.adaptiveThreshold(img_warp_gray, 255, 1, 1, 7, 2)
            img_adaptive_threshold = cv2.bitwise_not(img_adaptive_threshold)
            img_adaptive_threshold = cv2.medianBlur(img_adaptive_threshold, 3)

            image_array = ([img, gray_image, img_threshold, contour_image],
                           [img_big_contour, img_warp_colored, img_warp_gray, img_adaptive_threshold])

        else:
            image_array = ([img, gray_image, img_threshold, contour_image],
                           [blank_image, blank_image, blank_image, blank_image])

        #ラベル
        labels = [["Original", "Gray", "Threshold", "Contours"],
                  ["Biggest Contour", "Warp Perspective", "Warp Gray", "Adaptive Threshold"]]

        stacked_image = utils.stack_images(image_array, 0.75, labels)
        cv2.imshow("Result", stacked_image)

        key = cv2.waitKey(30)
        if key != -1:
            filter_name = chr(key)
            if filter_name == 's':
                cv2.rectangle(stacked_image,
                              ((int(stacked_image.shape[1] / 2) - 230), int(stacked_image.shape[0] / 2) + 50),
                              (1100, 350), (0, 255, 0), cv2.FILLED)
                if img_warp_colored is not None:
                    cv2.imwrite("Scanned/output" + str(count) + ".jpg", img_warp_colored)
                    cv2.putText(stacked_image, "Scan Saved",
                                (int(stacked_image.shape[1] / 2) - 200, int(stacked_image.shape[0] / 2)),
                                cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
                else:
                    cv2.putText(stacked_image, "Document not found",
                                (int(stacked_image.shape[1] / 2) - 200, int(stacked_image.shape[0] / 2)),
                                cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)

                cv2.imshow('結果', stacked_image)
                cv2.waitKey(300)
                count += 1
            elif filter_name == 'q':
                cv2.destroyAllWindows()
                exit()


if __name__ == "__main__":
    scan_document()
