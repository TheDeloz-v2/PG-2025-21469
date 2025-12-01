import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from .predictor import ModelPredictor


class ModelValidator:
    """
    Validador para evaluar predicciones de modelos contra datos reales
    """
    
    def __init__(
        self,
        tesla_data_path: str = 'data/raw/tesla-stock.csv',
        stock_dataset_path: str = 'src/data/processed/fusion/dataset_stock.csv',
        sentiment_dataset_path: str = 'src/data/processed/fusion/dataset_sentiment.csv',
        models_dir: str = 'models'
    ):
        self.tesla_data_path = tesla_data_path
        self.stock_dataset_path = stock_dataset_path
        self.sentiment_dataset_path = sentiment_dataset_path
        self.models_dir = models_dir
        
        # Cargar datos
        self.tesla_data = None
        self.df_stock = None
        self.df_sentiment = None
        
        # Resultados
        self.predicciones = []
        self.resultados = []
    
    def get_next_available_date(self, target_date: str) -> Optional[str]:
        """
        Encuentra la siguiente fecha disponible en los datos
        
        Args:
            target_date: Fecha objetivo en formato 'YYYY-MM-DD'
            
        Returns:
            Fecha disponible más cercana o None
        """
        if self.tesla_data is None:
            raise ValueError("Tesla data no cargada. Llamar load_data() primero.")
        
        target_dt = pd.to_datetime(target_date)
        available = self.tesla_data[self.tesla_data['Date'] >= target_dt].sort_values('Date')
        if len(available) > 0:
            return available.iloc[0]['Date'].strftime('%Y-%m-%d')
        return None
    
    def load_data(self, cutoff_date: Optional[str] = None):
        """
        Carga todos los datos necesarios
        
        Args:
            cutoff_date: Fecha de corte para datasets (YYYY-MM-DD)
        """
        # Cargar datos de Tesla
        self.tesla_data = pd.read_csv(self.tesla_data_path)
        self.tesla_data['Date'] = pd.to_datetime(self.tesla_data['Date'])
        self.tesla_data = self.tesla_data.sort_values('Date')
        
        # Normalizar columna Close
        if 'close' in self.tesla_data.columns:
            self.tesla_data = self.tesla_data.rename(columns={'close': 'Close'})
        
        # Cargar datasets de features
        self.df_stock = pd.read_csv(self.stock_dataset_path)
        self.df_sentiment = pd.read_csv(self.sentiment_dataset_path)
        
        # Convertir fechas
        for df in [self.df_stock, self.df_sentiment]:
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Aplicar cutoff si se especifica
        if cutoff_date:
            cutoff_dt = pd.to_datetime(cutoff_date)
            self.df_stock = self.df_stock[self.df_stock['Date'] <= cutoff_dt].copy()
            self.df_sentiment = self.df_sentiment[self.df_sentiment['Date'] <= cutoff_dt].copy()
    
    def get_base_price(self, date: str) -> Tuple[float, str]:
        """
        Obtiene el precio base para una fecha
        
        Args:
            date: Fecha en formato 'YYYY-MM-DD'
            
        Returns:
            Tupla (precio, fecha_efectiva)
        """
        precio_array = self.tesla_data[self.tesla_data['Date'] == date]['Close'].values
        
        if len(precio_array) == 0:
            prev_date = pd.to_datetime(date) - pd.Timedelta(days=1)
            prev_date_str = prev_date.strftime('%Y-%m-%d')
            precio_array = self.tesla_data[self.tesla_data['Date'] == prev_date_str]['Close'].values
            
            if len(precio_array) == 0:
                raise ValueError(f"No se encontró precio para {date} ni {prev_date_str}")
            
            return float(precio_array[0]), prev_date_str
        
        # Retornar dia anterior como fecha efectiva
        effective_date = (pd.to_datetime(date) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        return float(precio_array[0]), effective_date
    
    def predict_with_model(
        self,
        model_name: str,
        dataset: pd.DataFrame,
        fecha_objetivo: str,
        descripcion: str,
        expected_acc: float,
        horizon: str
    ) -> Optional[Dict]:
        """
        Genera predicción con un modelo
        
        Args:
            model_name: Nombre del modelo (e.g., 'sentiment_short')
            dataset: DataFrame con features
            fecha_objetivo: Fecha objetivo de predicción
            descripcion: Descripción del modelo
            expected_acc: Precisión esperada del modelo
            horizon: Horizonte temporal
            
        Returns:
            Diccionario con predicción o None si hay error
        """
        try:
            model_path = f"{self.models_dir}/{model_name}_model.pth"
            scalers_path = f"{self.models_dir}/{model_name}_scalers.pkl"
            
            predictor = ModelPredictor(model_path, scalers_path)
            prediction = predictor.predict_from_dataframe(dataset)
            
            pred_data = {
                'modelo': model_name,
                'descripcion': descripcion,
                'horizon': horizon,
                'fecha_objetivo': fecha_objetivo,
                'retorno_predicho': prediction['predicted_return'],
                'direccion_predicha': 'UP' if prediction['predicted_return'] > 0 else 'DOWN',
                'expected_acc': expected_acc
            }
            
            return pred_data
            
        except Exception as e:
            print(f"Error: {str(e)}")
            return None
    
    def validate_prediction(
        self,
        pred: Dict,
        precio_base: float
    ) -> Optional[Dict]:
        """
        Valida una predicción contra datos reales
        
        Args:
            pred: Diccionario con predicción
            precio_base: Precio base para calcular retorno
            
        Returns:
            Diccionario con validación o None si no hay datos
        """
        fecha_obj = pred['fecha_objetivo']
        precio_obj_array = self.tesla_data[self.tesla_data['Date'] == fecha_obj]['Close'].values
        
        if len(precio_obj_array) == 0:
            print(f"{pred['descripcion']}: No hay datos para {fecha_obj}")
            return None
        
        precio_objetivo = float(precio_obj_array[0])
        retorno_real = (precio_objetivo - precio_base) / precio_base
        direccion_real = 'UP' if retorno_real > 0 else 'DOWN'
        
        acierto = (pred['direccion_predicha'] == direccion_real)
        error_abs = abs(pred['retorno_predicho'] - retorno_real)
        
        resultado = {
            **pred,
            'precio_base': precio_base,
            'precio_objetivo': precio_objetivo,
            'retorno_real': retorno_real,
            'direccion_real': direccion_real,
            'acierto': acierto,
            'error_abs': error_abs
        }
        
        return resultado
    
    def run_validation(
        self,
        simulation_date: str = '2025-04-17',
        cutoff_date: str = '2025-04-16',
        top_models: Optional[List[Dict]] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Ejecuta validación completa
        
        Args:
            simulation_date: Fecha de simulación
            cutoff_date: Fecha de corte para datos
            top_models: Lista de modelos a validar (None = usar predeterminados)
            verbose: Imprimir salida detallada
            
        Returns:
            Diccionario con resultados de validación
        """
        if verbose:
            print("=" * 80)
            print("VALIDACION COMPLETA - MEJORES MODELOS POR HORIZONTE")
            print("=" * 80)
            print()
        
        # Cargar datos
        self.load_data(cutoff_date)
        
        if verbose:
            print(f"Datos cargados: {len(self.tesla_data)} registros")
            print(f"Desde: {self.tesla_data['Date'].min().strftime('%Y-%m-%d')}")
            print(f"Hasta: {self.tesla_data['Date'].max().strftime('%Y-%m-%d')}")
            print()
        
        # Obtener precio base
        precio_base, fecha_base = self.get_base_price(simulation_date)
        
        if verbose:
            print(f"Precio base ({fecha_base}): ${precio_base:.2f}")
            print()
        
        # Ajustar fechas objetivo
        fecha_short_real = self.get_next_available_date(
            (pd.to_datetime(simulation_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        )
        fecha_medium_real = '2025-04-24'
        fecha_long_real = '2025-05-16'
        
        # Generar predicciones
        if verbose:
            print(f"Datasets hasta {cutoff_date}")
            print(f"Stock: {len(self.df_stock)} registros")
            print(f"Sentiment: {len(self.df_sentiment)} registros")
            print()
        
        self.predicciones = []
        
        for modelo_info in top_models:
            if verbose:
                print(f"{modelo_info['descripcion']} ({modelo_info['horizon']})")
                print(f"Expected Acc: {modelo_info['expected_acc']:.2f}%")
            
            pred_data = self.predict_with_model(
                modelo_info['name'],
                modelo_info['dataset'],
                modelo_info['fecha_objetivo'],
                modelo_info['descripcion'],
                modelo_info['expected_acc'],
                modelo_info['horizon']
            )
            
            if pred_data:
                self.predicciones.append(pred_data)
                
                if verbose:
                    direction = "UP" if pred_data['retorno_predicho'] > 0 else "DOWN"
                    print(f"Prediccion: {pred_data['retorno_predicho']:+.2%} ({direction})")
                    print()
        
        if not self.predicciones:
            print("Error: No se generaron predicciones")
            return {}
        
        # Validar con datos reales
        self.resultados = []
        
        for pred in self.predicciones:
            resultado = self.validate_prediction(pred, precio_base)
            
            if resultado:
                self.resultados.append(resultado)
                
                if verbose:
                    self._print_result_box(resultado)
        
        # Calcular métricas
        metrics = self._calculate_metrics()
        
        if verbose:
            print("=" * 80)
            print("METRICAS CONSOLIDADAS")
            print("=" * 80)
            print()
            self._print_metrics(metrics)
            print()
            
            print("=" * 80)
            print("COMPARACION EXPECTATIVA vs REALIDAD")
            print("=" * 80)
            print()
            self._print_comparison_table()
        
        return {
            'predicciones': self.predicciones,
            'resultados': self.resultados,
            'metrics': metrics,
            'precio_base': precio_base,
            'simulation_date': simulation_date,
            'cutoff_date': cutoff_date
        }
    
    def _print_result_box(self, resultado: Dict):
        """Imprime resultado en formato de caja"""
        pred = resultado
        precio_base = resultado['precio_base']
        precio_objetivo = resultado['precio_objetivo']
        retorno_real = resultado['retorno_real']
        direccion_real = resultado['direccion_real']
        acierto = resultado['acierto']
        error_abs = resultado['error_abs']
        fecha_obj = resultado['fecha_objetivo']
        
        print("+" + "-"*78 + "+")
        print(f"| {pred['descripcion']:<40} ({pred['horizon']:<15}) |")
        print("+" + "-"*78 + "+")
        print(f"| Expected Acc (training): {pred['expected_acc']:.2f}%{' '*39} |")
        print(f"| Fecha: {fecha_obj}{' '*62} |")
        print(f"| Precio: ${precio_base:.2f} -> ${precio_objetivo:.2f}{' '*51} |")
        print("+" + "-"*78 + "+")
        print(f"| PREDICCION: {pred['retorno_predicho']:+7.2%} ({pred['direccion_predicha']:4}){' '*45} |")
        print(f"| REALIDAD:   {retorno_real:+7.2%} ({direccion_real:4}){' '*45} |")
        print("+" + "-"*78 + "+")
        
        if acierto:
            print(f"| DIRECCION ACERTADA{' '*59} |")
        else:
            print(f"| DIRECCION FALLIDA{' '*60} |")
        
        print(f"| Error: {error_abs:.4f} ({error_abs*100:.2f} pp){' '*44} |")
        print("+" + "-"*78 + "+")
        print()
    
    def _calculate_metrics(self) -> Dict:
        """Calcula métricas consolidadas"""
        aciertos = sum(r['acierto'] for r in self.resultados)
        total = len(self.resultados)
        dir_acc = (aciertos / total * 100) if total > 0 else 0
        mae = np.mean([r['error_abs'] for r in self.resultados]) if self.resultados else 0
        
        return {
            'total_modelos': total,
            'aciertos': aciertos,
            'directional_accuracy': dir_acc,
            'mae': mae
        }
    
    def _print_metrics(self, metrics: Dict):
        """Imprime métricas consolidadas"""
        print(f"RESULTADOS GLOBALES:")
        print("-" * 80)
        print(f"Modelos validados: {metrics['total_modelos']}/3")
        print(f"Directional Accuracy: {metrics['directional_accuracy']:.2f}% "
              f"({metrics['aciertos']}/{metrics['total_modelos']} aciertos)")
        print(f"Error promedio (MAE): {metrics['mae']*100:.2f} puntos porcentuales")
        print()
        
        dir_acc = metrics['directional_accuracy']
        if dir_acc >= 60:
            print(f"Evaluacion: EXCELENTE (>60%)")
        elif dir_acc >= 50:
            print(f"Evaluacion: BUENO (>50%, mejor que azar)")
        elif dir_acc > 0:
            print(f"Evaluacion: BAJO pero positivo")
        else:
            print(f"Evaluacion: SIN ACIERTOS")
    
    def _print_comparison_table(self):
        """Imprime tabla de comparación"""
        print("+" + "-"*78 + "+")
        print(f"| {'MODELO':<35} | {'ESPERADO':<10} | {'REAL':<10} | {'DIFERENCIA':<15} |")
        print("+" + "-"*78 + "+")
        
        for r in self.resultados:
            esperado = r['expected_acc']
            obtenido = 100.0 if r['acierto'] else 0.0
            diff = obtenido - esperado
            diff_str = f"{diff:+.2f} pp"
            status = "OK" if diff >= 0 else "BAJO"
            
            print(f"| {r['descripcion']:<35} | {esperado:>8.2f}% | {obtenido:>8.1f}% | "
                  f"{diff_str:<12} {status:<3} |")
        
        print("+" + "-"*78 + "+")
        
        avg_expected = np.mean([r['expected_acc'] for r in self.resultados])
        dir_acc = self._calculate_metrics()['directional_accuracy']
        diff_avg = dir_acc - avg_expected
        
        print(f"| {'PROMEDIO':<35} | {avg_expected:>8.2f}% | {dir_acc:>8.2f}% | "
              f"{diff_avg:+10.2f} pp    |")
        print("+" + "-"*78 + "+")
        print()
