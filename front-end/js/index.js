/* index.htmlのjsはココ */

//設定
// test_environment = 'https://private-anon-0b73dcc383-aichangesfaces.apiary-mock.com/v1/faceAnalysis'
// real_environment = 'https://blooming-caverns-18971.herokuapp.com/v1/faceAnalysis'
// apiurl = real_environment

// 変数定義
var image_url

// ローディング
$loading = $(".loading");

// 画像のアップロード
$('#file').change(function (e){

    var img = new Image();
    var reader = new FileReader();
    var file = this.files[0];
    var datasize = this.files[0].size;
    
    if (!file.type.match(/^image\/(png|jpg|jpeg)$/)){
        alert("対応画像ファイルは[png|jpg|jpeg]のみです。");
        return false;
    }

    reader.onload = function(event){

        $loading.removeClass("is-hide");

        img.onload = function(){
            var data = {data:img.src.split(',')[1]};

            function post_pic(){
            var result = $.ajax({
                type: 'POST',
                url: '../pic.php',
                data: data,
                async: false,
            }).responseText;

            setTimeout(function(){
            $.when(result).done(function(){
                $loading.addClass("is-hide");
            });
            },400);

            return result;

            }

            image_url = post_pic()
            console.log(image_url)

        };
        img.src = event.target.result;

    };
    
    reader.readAsDataURL(file);

    // 画像を表示
    var input = $(this),
    numFiles = input.get(0).files ? input.get(0).files.length : 1,
    label = input.val().replace(/\\/g, '/').replace(/.*\//, '');
    input.parent().parent().next(':text').val(label);

    $loading.addClass("is-hide");

});

// データ送信API
$('#start').click(function(){
    //alert(this.id);

    if (image_url == null) {
        alert("画像を選択してください。");
        return false;
    }

    $loading.removeClass("is-hide");

    var nexturl = "http://haiku-generator.tokyo/result.php?url=" +image_url
    document.location.href = nexturl;

 //    // ここからAPI送信
	// var request = new XMLHttpRequest();
	// var fullname = document.getElementById('fullname').value;
	// request.open('POST', apiurl);
	// request.setRequestHeader('Content-Type', 'application/json');
	// var body = {
	//     'image_url': image_url,
	//     'analysis_type': 'seasoning'
	// };
	// request.send(JSON.stringify(body));
 //    request.onreadystatechange = function () {
 //      if (this.readyState === 4) {

 //        try {

 //            // 失敗するかもしれないメソッド
 //            console.log(this.responseText);
 //            var body = JSON.parse(this.responseText);
 //            var nexturl = "result.html?fullname="+fullname+"&url="+image_url+"&"+body[0].attribute_name1+"="+body[0].attribute1+"&"+body[1].attribute_name2+"="+body[1].attribute2+"&"+body[2].attribute_name3+"="+body[2].attribute3
 //            console.log(nexturl);

 //            //データSlack送信
 //                var slackurl = 'https://slack.com/api/chat.postMessage';

 //                    var data = {
 //                        token: token_info,
 //                        channel: '#apply',
 //                        username: '顔診断されました',
 //                        as_user: 'false',
 //                        icon_url: 'http://lorempixel.com/48/48',
 //                        text: 'Name: '+fullname+'\n'+'url: '+image_url+'\n'+'しお顔度: '+body[0].attribute1+'\n'+'しょうゆ顔度: '+body[1].attribute2+'\n'+'ソース顔度: '+body[2].attribute3
 //                    };

 //                    $.ajax({
 //                        type: 'GET',
 //                        url: slackurl,
 //                        data: data,
 //                        async: false
 //                    });

 //            document.location.href = nexturl;

 //        } catch(e) {

 //            // 失敗した場合のメソッド
 //            $loading.addClass("is-hide");
 //            alert("写真をうまく読み込めませんでした。顔が写っていない、２人以上の顔が写っている、顔が読み込めない可能性があります。他の写真をアップロードしてください。");
 //        }


 //     }
 //    };

	// return false;

});

//ページの頭に戻る処理
$(function() {
  $('.btn-top').click(function() {
    // スクロール
    $('html,body').animate({scrollTop: 0}, 500, 'swing');
  });
});


//画像選択のフォーム作成
// $(document).on('change', ':file', function() {
//     var input = $(this),
//     numFiles = input.get(0).files ? input.get(0).files.length : 1,
//     label = input.val().replace(/\\/g, '/').replace(/.*\//, '');
//     input.parent().parent().next(':text').val(label);
// });

// 参考 http://blog2.gods.holy.jp/?eid=189
