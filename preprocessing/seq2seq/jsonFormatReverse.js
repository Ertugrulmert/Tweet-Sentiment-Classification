const fs = require('fs');

tweets = []
fs.readFile('./json_result/result.json', 'utf8', (err, data) => {
  data = JSON.parse(data)

  data = data.sort((a,b) => a.index - b.index)
  let i = 0
    data.forEach(obj => {
      tweets.push(obj.output.join(" "))
    })

    fs.writeFile('./raw_result/result.txt', tweets.join("\n"), err => {
      if (err) {
        console.error(err);
      }
    });
});