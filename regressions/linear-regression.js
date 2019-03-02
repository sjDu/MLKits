const tf = require('@tensorflow/tfjs')
const _ = require('lodash')

class LinearRegression {
    constructor(features, labels, options) {
        this.features = this.processFeatures(features)
        this.labels = tf.tensor(labels)



        this.options = Object.assign({
            learningRate: 0.1,
            iterations: 1000,
        }, options) 

        // features now already has an addon column 11111 so we don't add 1
        // [b, m1, m2, m3, ..., m_n]
        this.weights = tf.zeros([this.features.shape[1],1])

        // this.m = 0
        // this.b = 0
    }

    gradientDescent() {
        const currentGuesses = this.features.matMul(this.weights)
        const differences = currentGuesses.sub(this.labels)

        const slopes = this.features
            .transpose()
            .matMul(differences)
            .div(this.features.shape[0])

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
        for( let i = 0; i < this.options.iterations; i++) {
            this.gradientDescent()
        }
    }

    test(testFeatures, testLabels) {
        testFeatures = this.processFeatures(testFeatures)
        testLabels = tf.tensor(testLabels)


        const predictions = testFeatures.matMul(this.weights)

        // Sum of squares of residuals
        const res = testLabels
            .sub(predictions)
            .pow(2)
            .sum()
            .get()

        // Total sum of squares
        const tot = testLabels.sub(testLabels.mean())
            .pow(2)
            .sum()
            .get()

        return 1 - res / tot
    }

    processFeatures(features) {
        features = tf.tensor(features)

        if(this.mean && this.variance){
            features = features.sub(this.mean).div(this.variance.pow(0.5))
        } else {
            features = this.standardize(features)
        }

        features = tf.ones([features.shape[0],1]).concat(features, 1)


        return features

    }

    standardize(features) {
        const { mean, variance } = tf.moments(features, 0) 

        this.mean = mean
        this.variance = variance

        return features.sub(mean).div(variance.pow(0.5))

    }
}

module.exports = LinearRegression