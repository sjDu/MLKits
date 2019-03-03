const tf = require('@tensorflow/tfjs')
const _ = require('lodash')

class LogisticRegression {
    constructor(features, labels, options) {
        const { mean, variance } = tf.moments(tf.tensor(features), 0)
        this.mean = mean
        this.variance = variance 

        this.features = this.processFeatures(features)
        this.labels = tf.tensor(labels)
        this.costHistory = []



        this.options = Object.assign({
            learningRate: 0.1,
            iterations: 1000,
            decisionBoundary: 0.5
        }, options) 

        // features now already has an addon column 11111 so we don't add 1
        // [b, m1, m2, m3, ..., m_n]
        this.weights = tf.zeros([this.features.shape[1], this.labels.shape[1]])

        // this.m = 0
        // this.b = 0
    }

    gradientDescent(features, labels) {
        const currentGuesses = features.matMul(this.weights).sigmoid()
        const differences = currentGuesses.sub(labels)

        const slopes = features
            .transpose()
            .matMul(differences)
            .div(features.shape[0])

        this.weights = this.weights.sub(
            slopes.mul(this.options.learningRate)
        )
    }

    gradientDescent_() {
        // const currentGuessForMPG = this.features.map(row => {
        //     return this.m * row[0] + this.b 
        // })

        // const bSlope = _.sum(currentGuessForMPG.map((guess, i) => {
        //     return guess - this.labels[i][0]
        // })) * 2 / this.features.length

        // const mSlope = _.sum(currentGuessForMPG.map((guess, i) => {
        //     return -1 * this.features[i][0] * (this.labels[i][0] - guess)
        // })) * 2 / this.features.length

        // this.m = this.m - this.options.learningRate * mSlope
        // this.b  = this.b - this.options.learningRate * bSlope

    }

    train() {
        const batchQuantity = Math.floor(
            this.features.shape[0] / this.options.batchSize
        )

        for( let i = 0; i < this.options.iterations; i++) {
            for (let j = 0; j < batchQuantity; j++) {
                const { batchSize } = this.options
                const startIndex = j * batchSize
                const featureSlice = this.features.slice(
                    [startIndex, 0],
                    [batchSize, -1]
                )
                const labelSlice = this.labels.slice(
                    [startIndex, 0],
                    [batchSize, -1]
                )

                this.gradientDescent(featureSlice, labelSlice)
            }
            this.recordCost()
            this.updateLearningRate() 
        }
    }

    test(testFeatures, testLabels) {

        const predictions = this.predict(testFeatures)
        testLabels = tf.tensor(testLabels).argMax(1)

        const incorrect =  predictions
            .notEqual(testLabels)
            .sum()
            .get()

        return (predictions.shape[0] - incorrect) / predictions.shape[0]
    }

    processFeatures(features) {
        features = tf.tensor(features)
        features = this.standardize(features)
        features = tf.ones([features.shape[0],1]).concat(features, 1)


        return features

    }

    standardize(features) {
        const { mean, variance } = this

        return features.sub(mean).div(variance.pow(0.5))

    }

    recordCost() {
        const guesses = this.features.matMul(this.weights).softmax()

        const termOne = this.labels
            .transpose()
            .matMul(guesses.log())

        const termTwo = this.labels
            .mul(-1)
            .add(1)
            .transpose()
            .matMul(
                guesses
                    .mul(-1)
                    .add(1)
                    .log()
            )


        const cost = termOne
            .add(termTwo)
            .div(this.features.shape[0])
            .mul(-1)
            .get(0, 0)
            
        this.costHistory.unshift(cost)
    }

    updateLearningRate() {
        if (this.costHistory.length < 2) {
            return
        }

        if (this.costHistory[0] > this.costHistory[1]) {
            this.options.learningRate /= 2
        } else {
            this.options.learningRate *= 1.05
        }
    }

    predict(observations) {
        return this.processFeatures(observations)
            .matMul(this.weights)
            .softmax()
            .argMax(1)
    }
}

module.exports = LogisticRegression

// sigmoid => marginal probability
// softmax => conditional probability