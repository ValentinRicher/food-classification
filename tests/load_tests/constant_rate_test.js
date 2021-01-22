import http from 'k6/http';
import encoding from 'k6/encoding';
import { sleep } from 'k6';

import { Rate } from 'k6/metrics';
const myFailRate = new Rate('failed requests');

const img = open('/images/812047.jpg', 'b');
let b64_img = encoding.b64encode(img);


export let options = {
    scenarios: {
      constant_request_rate: {
        executor: 'constant-arrival-rate',
        rate: 20,
        timeUnit: '1s',
        duration: '10m',
        preAllocatedVUs: 100,
        maxVUs: 200,
      },
    },
    thresholds: {
        'failed requests': ['rate<0.1'], // threshold on a custom metric
        // http_req_duration: ['p(95)<1000'], // threshold on a standard metric
      },
  };



export default function() {
    // const url = "http://40.89.187.133:80/api/v1/service/food-service/score"; // AKS_1
    // let token = "KrxNUPTQcvMJIlU1qatARs8xvTGqGOiI"
    // const url = "http://51.103.33.230:80/api/v1/service/food-service-2/score"; // AKS_2    
    // let token = "IvsZqj7FgNu2EwNigkrsxK209PYYpPjo"
    const url = "http://51.103.82.254:80/api/v1/service/food-service-3/score" // AKS_3
    let token = "NO1ehdY3debEU3opEJOCC55VlIyP9it7"
    let data = {columns:["image"],index:[0],data:b64_img};
    var params = {};
    params["headers"] = {"Content-Type": "application/json; format=pandas-split"}
    params["headers"]["Authorization"] = `Bearer ${token}`
    let res = http.post(url, JSON.stringify(data), params);
    // console.log(res.body);
    myFailRate.add(res.status !== 200);
}