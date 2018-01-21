/* result.htmlのjsはココ */


// //ツイートBOX
// var href = encodeURIComponent(window.location.href);
// document.getElementById("tweet-box").innerHTML = "<a href='https://twitter.com/intent/tweet?text=HaikuGeneratorで、手持ちの画像を俳句にしてみよう！私の俳句はこちら->&url="+href+"&hashtags=HaikuGenerator' rel='nofollow' onClick='window.open(encodeURI(decodeURI(this.href)),'twwindow','width=550, height=450, personalbar=0, toolbar=0, scrollbars=1'); return false;' class='btn btn-info btn-lg btn-block active btn-top'>結果をツイートする</a>";

// //ツイート送信
// !function(d,s,id){var js,fjs=d.getElementsByTagName(s)[0],p=/^http:/.test(d.location)?'http':'https';if(!d.getElementById(id)){js=d.createElement(s);js.id=id;js.src=p+'://platform.twitter.com/widgets.js';fjs.parentNode.insertBefore(js,fjs);}}(document, 'script', 'twitter-wjs');


//URLクエリパラメータをJSONに変換
var QueryString = {  
  parse: function(text, sep, eq, isDecode) {
    text = text || location.search.substr(1);
    sep = sep || '&';
    eq = eq || '=';
    var decode = (isDecode) ? decodeURIComponent : function(a) { return a; };
    return text.split(sep).reduce(function(obj, v) {
      var pair = v.split(eq);
      obj[pair[0]] = decode(pair[1]);
      return obj;
    }, {});
  },
  stringify: function(value, sep, eq, isEncode) {
    sep = sep || '&';
    eq = eq || '=';
    var encode = (isEncode) ? encodeURIComponent : function(a) { return a; };
    return Object.keys(value).map(function(key) {
      return key + eq + encode(value[key]);
    }).join(sep);
  },
};



var arraydata = QueryString.parse();

function floatFormat( number, n ) {
    var _pow = Math.pow( 10 , n ) ;

    return Math.round( number * _pow ) / _pow ;
}

    var name = arraydata.fullname
    console.log(name)
    $("#yourname").text(name);

    url = arraydata.url //グローバル変数に画像URLを付与
    console.log(url)
    $("#yourface").attr("src",url);



// 参考 http://blog2.gods.holy.jp/?eid=189


