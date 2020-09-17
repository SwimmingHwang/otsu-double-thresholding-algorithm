import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


# path = '/Users/young/test2.png'


def otsu(obj, src, imgName, minX, minY, maxX, maxY, paramList):
	"""
	삼중 임계값 오츄 알고리즘

	:param obj: Object name string to get
	:param src: Image numpy array
	:param imgName: File's name
	:param minX: Deep learning output bounding box info
	:param minY:
	:param maxX:
	:param maxY:
	:param paramList: Image filter parameter
	:return:
	"""

	'''Bilateral Filter Parameter Setting '''
	# Input Image Red Channel 추출
	redImg = src[:, :, 2]  # opencv 는 b,g,r

	'''
	d – filtering시 고려할 주변 pixel 지름
	sigmaColor – Color를 고려할 공간. 숫자가 크면 멀리 있는 색도 고려함.
	sigmaSpace – 숫자가 크면 멀리 있는 pixel도 고려함.
	'''
	redImg = cv.bilateralFilter(redImg, paramList[0], paramList[1], paramList[2])
	padding = 3  # main object bounding box에서 테두리에 padding 넣기 (잘림방지)

	roi = np.zeros(redImg.shape[:2], np.uint8)
	roi[minY - padding:maxY + padding, minX - padding:maxX + padding] = 255
	roiImg = cv.bitwise_and(redImg, redImg, mask=roi)

	# 이미지 히스토그램
	hist = cv.calcHist([redImg], [0], roi, [256], [0, 256])
	plt.plot(hist)
	plt.xlim([0, 256])

	# 히스토그램 값 가져오기
	ax = plt.gca()
	line = ax.lines[0]
	print(line)
	graph = line.get_xydata().astype(int)
	plt.show()
	"""
	########### Otsu Algorihm ######### 
	삼중 임계값 얻어오기
	"""
	L = 256
	maxVar = -1
	maxT1 = 0
	maxT2 = 0
	uG = 0

	''' 처음에만 전체 평균 저장하고 있기 '''
	sumG = 0
	for i in range(0, L):
		sumG += i * graph[i][1]
	uG = (1 / np.sum(graph[0:L - 1 + 1], axis=0)[1]) * sumG
	print("Total Mean : ", uG)

	sumA = 0 * graph[0][1]
	wA = graph[0][1]
	if wA != 0:
		uA = (1 / wA) * sumA
	else:
		uA = 0

	for t1 in range(1, L - 3 + 1):
		sumA += t1 * graph[t1][1]
		wA += graph[t1][1]
		if wA != 0:
			uA = (1 / wA) * sumA

		sumB = 0
		wB = 0
		uB = 0

		sumC = 0
		wC = np.sum(graph[t1 + 1:L - 1 + 1], axis=0)[1]
		for i in range(t1 + 1, L - 1 + 1):
			sumC += i * graph[i][1]
		uC = 0
		if wC != 0:
			uC = (1 / wC) * sumC

		for t2 in range(t1 + 1, L - 2 + 1):
			sumB += t2 * graph[t2][1]
			wB += graph[t2][1]
			if wB != 0:
				uB = (1 / wB) * sumB

			sumC -= t2 * graph[t2][1]
			wC -= graph[t2][1]
			if wC != 0:
				uC = (1 / wC) * sumC

			# var(t1,t2) 크면 클 수록 좋음.
			var = sumA * (uA - uG) ** 2 + sumB * (uB - uG) ** 2 + sumC * (uC - uG) ** 2

			# 가장 큰 값을 생성한 쌍 구하기
			if maxVar < var:
				maxVar = var
				maxT1 = t1
				maxT2 = t2


	print(maxT1, maxT2)

	# 얻은 임계값으로 이미지 삼진화
	for y in range(minY - padding, maxY + padding):
		for x in range(minX - padding, maxX + padding):
			if roiImg[y][x] < maxT1:
				roiImg[y][x] = 0
			else:
				if roiImg[y][x] >= maxT2:
					roiImg[y][x] = 255
				else:
					roiImg[y][x] = 128

	# 얻은 삼진화 영상에서 세번째 구역에 속하는 픽셀로 이루어진 외곽선 구하기
	contourImg = roiImg.copy()
	ret, thresh = cv.threshold(contourImg, 200, 255, 0)
	image, contours, hierachy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


	# contours 여러개 나올때 제일 사이즈 큰 애로 하게 해야함 and Reflection point 무시해줘야 함.
	# 2-1 ) reflection point 제외한 큰 contour 선택
	minIdx = -1
	minSize = 99999999
	for idx, con in enumerate(contours):
		(x, y, w, h) = cv.boundingRect(contours[idx])
		distance = ((minX + maxX) / 2 - x) ** 2 + ((minY + maxY) / 2 - y) ** 2
		if minSize > distance:
			# print(con)
			minIdx = idx
			minSize = distance

	# 2-2 ) 사이즈가 제일 큰 contour 선택
	maxIdx = -1
	maxSize = 0
	for idx, con in enumerate(contours):
		if minIdx != idx:
			(x, y, w, h) = cv.boundingRect(contours[idx])
			if maxSize < w + h:
				# print(con)
				maxIdx = idx
				maxSize = w + h

	if len(contours) >= 2:
		image = cv.drawContours(src, contours, -1, (0, 255, 0), 1)
		(x, y, w, h) = cv.boundingRect(contours[maxIdx])
		print(x, y, w, h)

		cv.rectangle(image, (minX, minY), (maxX, maxY), (0, 0, 255), 1)
		cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
	else:
		return None

	# 크기작은 케이스 예외처리 하기
	if w <= padding or h <= padding:
		return None

	return image
