package it.unibo.neat

import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._
import org.encog.ml.MLRegression
import org.encog.ml.data.basic.BasicMLData

import java.io.{FileInputStream, ObjectInputStream}

class ScafiHopCountFromFile
    extends AggregateProgram
    with StandardSensors
    with ScafiAlchemistSupport
    with FieldUtils {
  private val howMany = 16.0 //helps to normalize nodes..
  private lazy val regression = {
    val fis = new FileInputStream("bestgene")
    val ois = new ObjectInputStream(fis)
    val user = ois.readObject.asInstanceOf[MLRegression]
    user
  }
  override def main(): Any = {
    rep(howMany) { data => {
      val elements = excludingSelf.reifyField(nbr(data)).values.toSeq :+ howMany
      val min = elements.min
      val normalized = min / howMany
      val sensor = if(node.has("source")) { 1.0 } else { 0.0 }
      val inputData = new BasicMLData(Array(normalized, sensor))
      val output = regression.compute(inputData).getData(0)
      node.put("output", Math.round(output * howMany))
      output
    }}
  }
}
