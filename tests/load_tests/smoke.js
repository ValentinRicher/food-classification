import http from 'k6/http';
import encoding from 'k6/encoding';
import { sleep } from 'k6';

const img = open('/images/812047.jpg', 'b');
let b64_img = encoding.b64encode(img);
// console.log(b64_img);

// smoke test
export let options = {
    vus: 1,
    duration: '1m',
};

export default function() {
    const url = "http://40.89.187.133:80/api/v1/service/food-service/score";
    let data = {columns:["image"],index:[0],data:b64_img};
    var params = {
        headers: {"Content-Type": "application/json; format=pandas-split", "Authorization": "Bearer KrxNUPTQcvMJIlU1qatARs8xvTGqGOiI"}
    };
    let res = http.post(url, JSON.stringify(data), params);
    // console.log(res.body);
}