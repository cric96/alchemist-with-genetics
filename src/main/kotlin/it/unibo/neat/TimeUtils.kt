package it.unibo.neat

import kotlin.system.measureTimeMillis


object TimeUtils {
    inline fun <T> measureTime(block : () -> T) : Pair<T, Long> {
        val result : T
        val time = measureTimeMillis {
            result = block()
        }
        return Pair(result, time)
    }
}