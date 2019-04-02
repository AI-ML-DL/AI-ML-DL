<%@ page language="java" contentType="text/html; charset=UTF-8"
	pageEncoding="UTF-8"%>
<%@ page import="java.io.*"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Naver Sentiment Movie Corpus Result</title>
</head>
<body>
	[영화 리뷰 감정 분석 결과]<br/>
	<%
	/* request 처리된 파라미터들의 인코딩 설정 */
	request.setCharacterEncoding("UTF-8");

	/* request 파라미터로 넘어온 값 확인 */
	String text = request.getParameter("text");
	String algorithm = request.getParameter("algorithm");
	System.out.println("입력 문장 : " + text);
	System.out.println("선택된 알고리즘 : " + algorithm);

	String output = "";

	/* Python 설치 바이너리 폴더 설정 */
	String pythonHome = "D:/Python36/";
	/* 알고리즘 학습 모델 및 python 파일 저장 폴더 설정 */
	String rscPath = "D:/MLBook/eclipse-workspace/SampleApplication/python/";
	/* 실행할 검증 구현 python 파일 위치 설정 */
	String pyFile = rscPath + "nsmc_evaluation.py";

	try {
		if (algorithm != null && text != null) {
			/* 외브 프로세스 실행을 위한 command 설정 */
			String[] command = { pythonHome + "/python", pyFile, algorithm, text };
			String line = null;

			/* 외부 프로세스 실행 */
			Process p = Runtime.getRuntime().exec(command);
			p.waitFor();

			/* 자식 프로세스의 입력 및 에러 스트림 저장 */
			BufferedReader br = new BufferedReader(
					new InputStreamReader(p.getInputStream()));
			BufferedReader stdError = new BufferedReader(
					new InputStreamReader(p.getErrorStream()));

			/* 입력 스트림을 읽어 내용 확인 및 output변수에 저장 */
			while ((line = br.readLine()) != null) {
				System.out.println(line);
				output += line + "\n";
			}
			br.close();

			/* 에러 스트림을 읽어 내용 확인 및 에러 메시지 저장 */
			while ((line = stdError.readLine()) != null) {
				System.out.println(line);
				output = "실행 시 에러가 발생하였습니다.";
			}
			stdError.close();
			p.destroy();
		} else {
			/* 파라미터 값이 없는 경우 에러 메시지 저장 */
			output = "알고리즘 또는 테스트 문장이 입력되지 않았습니다.";
		}
	} catch (Exception e) {
		e.printStackTrace();
	}
	/* 개행으로 나눠 String 배열에 저장 */
	String[] result = output.split("\\n");
	%>
	<!-- 선택된 알고리즘 및 결과 값 화면 출력 -->
	선택된 알고리즘 : <%=algorithm%><br/>
	<% 
		for (String str : result) {
	%>
	<%=str%><br/> <% } %>
	<!-- index.html 로 이동 -->
	<input type="button" value="뒤로 가기" onclick="location.href='index.html'" />
</body>
</html>