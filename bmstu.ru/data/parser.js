const table = document.querySelector('table.standart_table.progress_students.vertical_hover');
const trs = table.tBodies[0].querySelectorAll('tr:not(.tr-disabled)');
const data = [];

for (let i = 0; i < trs.length; ++i) {
    data.push([+trs[i].children[4].innerText,
               +trs[i].children[5].innerText,
               +trs[i].children[6].innerText,
              ]);
}

console.log(JSON.stringify(data))
