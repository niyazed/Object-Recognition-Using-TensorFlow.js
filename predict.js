

$("#image-selector").change(function(){

    let reader = new FileReader();

    reader.onload = function () {
        let dataUrl = reader.result;
        $("#selected-image").attr("src", dataUrl);
        $("#prediction-list").empty();
    }

    let file = $("#image-selector").prop('files')[0];
    reader.readAsDataURL(file);
});

let model;
(async function(){
    model = await tf.loadModel("https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json");
     //https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json
    $('.progress-bar').hide();
}) ();



$('#predict-button').click( async function() {
    let image = $('#selected-image').get(0);
    let tensor = tf.fromPixels(image)
        .resizeNearestNeighbor([224,224])
        .toFloat()
        
    //for VGG16
    // let meanImageNetRGB = tf.tensor1d([123.68,116.779,103.939]);
    // let pensor = tensor.sub(meanImageNetRGB)
    let offset = tf.scalar(127.5);
    let pensor = tensor.sub(offset)
        .div(offset)
        .expandDims();



    let predictions = await model.predict(pensor).data();
    let top5 = Array.from(predictions)
        .map(function (p, i) {
            return {
                probability: p,
                className: IMAGENET_CLASSES[i]
            };
        }).sort(function (a, b) {
            return b.probability - a.probability;
        }).slice(0, 5);

        $("#prediction-list").empty();
        top5.forEach(function (p) {
            $('#prediction-list').append(`<li>${p.className}: ${p.probability.toFixed(2)}`);
        });
});
