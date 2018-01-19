<?php


  $haiku = "テストです";
  $kigo = "テストだぞい";

  //urlを取得
  if(isset($_GET['url'])) {
      $picurl = $_GET['url'];
  }

    $option = [
        CURLOPT_RETURNTRANSFER => true, //文字列として返す
        CURLOPT_TIMEOUT        => 60, // タイムアウト時間
    ];

  	$url = "http://ec2-18-218-57-155.us-east-2.compute.amazonaws.com/haiku/?url=" . $picurl;

  js_console_log($url);

    $ch = curl_init($url);
    curl_setopt_array($ch, $option);
    $json    = curl_exec($ch);

  js_console_log($json);

  $json = mb_convert_encoding($json, 'UTF8', 'ASCII,JIS,UTF-8,EUC-JP,SJIS-WIN');
  $arr = json_decode($json,true);

  $haiku = $arr->haiku;
  $kigo = $arr->kigo;

function js_console_log( $array, $label = "", $trace = false, $trace_options = array() ) {
	if ( !is_array( $array ) ) {
		$array = array( $array );
	}

	if ( !empty( $label ) ) {
		$array = array(
			$label => $array,
		);
	}
	$array['__info']['message'] = 'output by js_console_log function';

	if ( $trace ) {

		if ( !isset( $trace_options[0] ) ) {
			$trace_options[0] = DEBUG_BACKTRACE_PROVIDE_OBJECT;
		}
		if ( !isset( $trace_options[1] ) ) {
			$trace_options[1] = 0;
		}
		$array['__info']['trace'] = PHP_VERSION_ID < 50400 ? debug_backtrace($trace_options[0]) : debug_backtrace($trace_options[0], $trace_options[1]);
	}

	$json = json_encode( $array );
	echo '<script type="text/javascript">';
	// IE対策
	echo "if (!('console' in window)) {";
	echo "    window.console = {};";
	echo "    window.console.log = function(str){   return str; };";
	echo "}";
	echo "console.log({$json})";
	echo '</script>';
}

?>

<!DOCTYPE html>
<html lang="ja">
	<head>
		<meta charset="UTF-8">
		<meta name="viewport" content="width=device-width">

		<meta name="twitter:card" content="summary_large_image" />
		<meta name="twitter:creator" content="@akihiko_1022" />
		<meta property="og:title" content="俳句ジェネレーター" />
		<meta property="og:description" content="画像を入力すると、画像に合った俳句を生成するサービスです。" />
		<meta property="og:url" content="http://haiku-generator.tokyo/" />

		<title>俳句ジェネレーター</title>

		<!-- CSSテンプレート読み込み -->
		<link href="https://code.jquery.com/ui/1.11.4/themes/ui-lightness/jquery-ui.css" rel="stylesheet">
		<link href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.min.css" rel="stylesheet">
		<link href="https://maxcdn.bootstrapcdn.com/font-awesome/4.5.0/css/font-awesome.min.css" rel="stylesheet">

		<!-- CSS読み込み -->
		<link href="css/common.css" rel="stylesheet">
		<link href="css/index.css" rel="stylesheet">

		<script>
		  (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
		  (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
		  m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
		  })(window,document,'script','https://www.google-analytics.com/analytics.js','ga');

		  ga('create', 'UA-112770320-1', 'auto');
		  ga('send', 'pageview');

		</script>

	</head>


	<body>

	
	<nav class="navbar navbar-default navbar-fixed-top">
		<div class="container-fluid">
			<div class="navbar-header">

			<a class="navbar-brand" href="index.html">俳句ジェネレーター</a>
			</div>
		
		</div>
	</nav>
	



		<div class="container">

			<div id="result">
				<div class="row text-center">
				<h2>俳句の生成結果</h2>
				</div>
				<hr>

				<div class="row">
					<div class="col-sm-6 text-center">
						<img id="yourface" src="<?php echo $picurl; ?>" width="300px" height="300px">
					</div>
					<div class="col-sm-6 text-center">
						<h3><?php echo($haiku); ?></h3>
						<h4>季語：<?php echo($kigo); ?></h4>
					</div>
					<br>

				</div>
			</div>
			
			<div class="row tryagain">
				<div class="col-xs-12 text-center">

				<a href="index.html" class="btn btn-success btn-lg btn-block active btn-top">もう一回やってみる</a>

				</div>
			</div>
	</div>

		<footer class="footer">
  			<div class="container">
    			<p class="text-muted text-center">(C) DeepFaceTeam. 2017. All Reights Reserved.</p>
  			</div>
		</footer>

		<!-- jquery読み込み -->
		<script src="https://code.jquery.com/jquery-1.11.3.min.js"></script>
		<script src="https://code.jquery.com/ui/1.11.4/jquery-ui.min.js"></script>

		<!-- bootstrap読み込み -->
		<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/js/bootstrap.min.js"></script>

		<!-- chart.js読み込み -->
		<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/1.0.2/Chart.min.js"></script>

		<!-- js読み込み -->
		<script src="js/token.js"></script>
		<script src="js/common.js"></script>
		<script src="js/result.js"></script>
	</body>
</html>
