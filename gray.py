#-*- coding:utf-8 -*-
import cv2
    
def main():
    for i in range(5380):
        i = i+1
        # 入力画像の読み込み
        img = cv2.imread('image_resize/%s.png' % str(i))
    
        # 方法2       
        gray2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
        # 結果を出力
        cv2.imwrite('./image_gray/'+'gray_%s.png' % str(i), gray2)
        print('Save','./image_gray/'+'gray_%s.png' % str(i), gray2)
    
if __name__ == "__main__":
    main()
