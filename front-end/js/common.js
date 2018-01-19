/* 全体共通のjsはココ */

// $(".loadingcss").click(function(){
// 	var $loading = $(".loading");
// 	$.ajax({
// 		url:"http://codepen.io/to-r/pen/bEeBrP.js",
// 		dataType:"json",
// 		beforeSend:function(){
// 			$loading.removeClass("is-hide");
// 		},
// 		success: function(data){
// 			$loading.addClass("is-hide");
// 			data.items.forEach(function(item){
// 				$("#list").append("<li>"+item.title+"</li>");
// 			});
// 		}
// 	});
// });