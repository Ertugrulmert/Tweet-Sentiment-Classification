const fs = require('fs');

json = []
fs.readFile('./raw_input/data.txt', 'utf8', (err, data) => {
  let i = 0
    data.split("\n").forEach(line => {
    i++
    const object = {
        tid: i,
        index: i,
        input: line.split(" "),
        output: line.split(" "),
    }

    json.push(object)   
    
  })

  fs.writeFile('./json_input/data.json', JSON.stringify(json), err => {
    if (err) {
      console.error(err);
    }
  });
});