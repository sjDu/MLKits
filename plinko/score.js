const outputs = []

const predict = 300

function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  // Ran every time a balls drops into a bucket
  outputs.push([dropPosition, bounciness, size, bucketLabel])

}

function runAnalysis() {
  // Write code here to analyze stuff

  const testSetSize = 50
  const [testSet, trainingSet] = splitDataset(minMax(outputs, 3), testSetSize)

  _.range(1, 15).forEach(k => {
    const accuracy = _.chain(testSet)
      .filter(testPoint => knn(trainingSet, _.initial(testPoint), k) === testPoint[3])
      .size()
      .divide(testSetSize)
      .value()
  
    console.log('for k of', k, ' the accuracy is ', accuracy)
    
  });


}

function knn(data, point, k) {

  // in future if we want to test a point which you don't know the label
  // you will not want to assign label so don't write `_.initial(point)` here
  // instead write `_.initial(testPoint)` outside
  return _.chain(data)
    .map(row => [
      distance(_.initial(row), point), 
      _.last(row)
    ])
    .sortBy(row => row[0])
    .slice(0, k)
    .countBy(row => row[1])
    .toPairs()
    .sortBy(row => row[1])
    .last()
    .first()
    .parseInt()
    .value()
}

function distance(pointA, pointB) {
  // pointA = [300, .5, 16], pointB = [ 350, .4, 12]
  return _.chain(pointA)
    .zip(pointB)
    .map(([a, b]) => (a - b) ** 2)
    .sum()
    .value() ** 0.5
  // return Math.abs(pointA - pointB)
}

function splitDataset(data, testCount) {
  const shuffled = _.shuffle(data)

  const testSet = _.slice(shuffled, 0, testCount)
  const trainingSet = _.slice(shuffled, testCount) 
  
  return [testSet, trainingSet]

}


// without mutation version
function minMax(data, featureCount) {

  const a = _.chain(data[0])
    .map((n, i) => {
      const column = data.map(row => row[i])
      if(i >= featureCount) {
        return column 
      }
      const min = _.min(column)
      const max = _.max(column)

      return _.map(column, it => (it - min)/(max - min))

    })
  	
   return _.zip(...a)

}

//Stephen's version
function minMax_(data, featureCount) {
  const clonedData = _.cloneDeep(data)

  for (let i = 0; i < featureCount; i++) {
    const column = clonedData.map(row => row[i])

    const min = _.min(column)
    const max = _.max(column)

    for (let j = 0; j < clonedData.length; j++) {
      clonedData[j][i] = (clonedData[j][i] - min)/(max - min)
    }

    return clonedData
  }
}