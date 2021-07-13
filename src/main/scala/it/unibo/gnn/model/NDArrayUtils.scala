package it.unibo.gnn.model

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

object NDArrayUtils {
  def nd[A: Numeric](elems: A*): INDArray = {
    Nd4j.create(Array(elems.map(elem => implicitly[Numeric[A]].toFloat(elem)).toArray))
  }
  def ndarray[B : Numeric](elems: Iterable[B]): INDArray = {
    Nd4j.create(Array(elems.map(elem => implicitly[Numeric[B]].toFloat(elem)).toArray))
  }

  def ndarray[B : Numeric](elems: Array[B]): INDArray = {
    Nd4j.create(Array(elems.map(elem => implicitly[Numeric[B]].toFloat(elem))))
  }
  def concat(ndarrays : INDArray*) : INDArray = {
    val flatten : Array[Float] = ndarrays.map { node => node.toDoubleVector.map(_.toFloat)} reduce { _ ++ _ }
    Nd4j.create(Array(flatten))
  }
}
