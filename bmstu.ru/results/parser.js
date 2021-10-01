const table = document.querySelector('table.eu-table');
const data = [];
const trs = table.tBodies[0].children;
for (let i = 0; i < trs.length; ++i) {
    switch(trs[i].children[7].children[1].innerText) {
        case "Отл": data.push(5); break;
        case "Хор": data.push(4); break;
        case "Удов": data.push(3); break;
        default: data.push(2);
    }
}

console.log(JSON.stringify(data))
