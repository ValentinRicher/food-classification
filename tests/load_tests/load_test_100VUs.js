import http from 'k6/http';
import encoding from 'k6/encoding';
import { sleep } from 'k6';

import { Rate } from 'k6/metrics';
const myFailRate = new Rate('failed requests');

const img = open('/images/812047.jpg', 'b');
let b64_img = encoding.b64encode(img);
// console.log(b64_img);

// load test
export let options = {
    stages: [
        { duration: '5m', target: 100 }, // simulate ramp-up of traffic from 1 to 100 users over 5 minutes.
        { duration: '10m', target: 100 }, // stay at 100 users for 10 minutes
        { duration: '5m', target: 0 }, // ramp-down to 0 users
    ],
    thresholds: {
        'failed requests': ['rate<0.1'], // threshold on a custom metric
        http_req_duration: ['p(95)<1000'], // threshold on a standard metric
      },
};

export default function() {
    // const url = "http://40.89.187.133:80/api/v1/service/food-service/score"; // AKS_1
    const url = "http://51.103.33.230:80/api/v1/service/food-service-2/score"; // AKS_2
    let data = {columns:["image"],index:[0],data:b64_img};
    var params = {
        headers: {"Content-Type": "application/json; format=pandas-split", "Authorization": "Bearer IvsZqj7FgNu2EwNigkrsxK209PYYpPjo"}
    };
    let res = http.post(url, JSON.stringify(data), params);
    // console.log(res.body);
    console.log(res.status);
    myFailRate.add(res.status !== 200);
}