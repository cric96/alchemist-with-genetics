package it.unibo.genetics

interface EndCondition<G> {
    fun endWhen(genotype : G) : Boolean
}

object EndConditions {
    fun <G> after(times : Int) : Any = object : EndCondition<G> {
        var count = 0
        override fun endWhen(genotype: G): Boolean {
            count += 1
            return count < times
        }
    }
}