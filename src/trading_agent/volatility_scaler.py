import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


class VolatilityScaler:
    """
    Ajusta predicciones considerando escalamiento de volatilidad temporal
    """
    
    def __init__(self, base_volatility: Optional[float] = None, base_horizon: int = 1):
        """
        Inicializar escalador
        
        Args:
            base_volatility: Volatilidad base (si None, se calcula del dataset)
            base_horizon: Horizonte base en días (default: 1 día)
        """
        self.base_volatility = base_volatility
        self.base_horizon = base_horizon
        self.volatility_by_horizon = {}
        
    def fit(self, df: pd.DataFrame, target_prefix: str = 'target_return_t+'):
        """
        Calcular volatilidades observadas para cada horizonte
        
        Args:
            df: DataFrame con targets de diferentes horizontes
            target_prefix: Prefijo de columnas target
        """
        # Encontrar todas las columnas target
        target_cols = [c for c in df.columns if c.startswith(target_prefix)]
        
        for col in target_cols:
            # Extraer horizonte
            horizon = int(col.split('+')[1])
            
            # Calcular volatilidad observada
            vol = df[col].std()
            self.volatility_by_horizon[horizon] = vol
        
        # Establecer volatilidad base si no se proporcionó
        if self.base_volatility is None and self.base_horizon in self.volatility_by_horizon:
            self.base_volatility = self.volatility_by_horizon[self.base_horizon]
        
        print(f"Volatilidades observadas:")
        for horizon in sorted(self.volatility_by_horizon.keys()):
            vol = self.volatility_by_horizon[horizon]
            print(f"   {horizon:2}d: {vol:.4f} ({vol*100:.2f}%)")
        print()
        
        return self
    
    def get_scaling_factor(self, from_horizon: int, to_horizon: int) -> float:
        """
        Calcular factor de escalamiento entre dos horizontes
        
        Args:
            from_horizon: Horizonte origen (días)
            to_horizon: Horizonte destino (días)
            
        Returns:
            Factor de escalamiento (√(to/from))
        """
        return np.sqrt(to_horizon / from_horizon)
    
    def get_theoretical_volatility(self, horizon: int) -> float:
        """
        Calcular volatilidad teórica para un horizonte dado
        
        Args:
            horizon: Horizonte en días
            
        Returns:
            Volatilidad teórica (σ_base × √(horizon/base_horizon))
        """
        if self.base_volatility is None:
            raise ValueError("Base volatility not set. Call fit() first or provide base_volatility.")
        
        return self.base_volatility * self.get_scaling_factor(self.base_horizon, horizon)
    
    def get_observed_volatility(self, horizon: int) -> Optional[float]:
        """
        Obtener volatilidad observada para un horizonte
        
        Args:
            horizon: Horizonte en días
            
        Returns:
            Volatilidad observada o None si no está disponible
        """
        return self.volatility_by_horizon.get(horizon)
    
    def scale_prediction(
        self,
        prediction: float,
        source_horizon: int,
        target_horizon: int,
        method: str = 'theoretical'
    ) -> float:
        """
        Escalar predicción de un horizonte a otro
        
        Args:
            prediction: Predicción original
            source_horizon: Horizonte de la predicción original
            target_horizon: Horizonte objetivo
            method: 'theoretical' (√t) o 'observed' (ratios empíricos)
            
        Returns:
            Predicción escalada
        """
        if source_horizon == target_horizon:
            return prediction
        
        if method == 'theoretical':
            # Usar escalamiento teórico (√t)
            factor = self.get_scaling_factor(source_horizon, target_horizon)
        
        elif method == 'observed':
            # Usar ratios empíricos observados
            if source_horizon not in self.volatility_by_horizon or \
               target_horizon not in self.volatility_by_horizon:
                raise ValueError(f"No observed volatility for horizons {source_horizon} or {target_horizon}")
            
            vol_source = self.volatility_by_horizon[source_horizon]
            vol_target = self.volatility_by_horizon[target_horizon]
            factor = vol_target / vol_source
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'theoretical' or 'observed'")
        
        # Escalar manteniendo la dirección
        sign = np.sign(prediction)
        magnitude = abs(prediction)
        scaled_magnitude = magnitude * factor
        
        return sign * scaled_magnitude
    
    def get_confidence_interval(
        self,
        prediction: float,
        horizon: int,
        confidence: float = 0.95,
        use_observed: bool = True
    ) -> Tuple[float, float]:
        """
        Calcular intervalo de confianza para una predicción
        
        Args:
            prediction: Predicción puntual
            horizon: Horizonte de predicción (días)
            confidence: Nivel de confianza (default: 95%)
            use_observed: Si True, usa volatilidad observada; si False, teórica
            
        Returns:
            Tupla (lower_bound, upper_bound)
        """
        # Obtener volatilidad
        if use_observed and horizon in self.volatility_by_horizon:
            vol = self.volatility_by_horizon[horizon]
        else:
            vol = self.get_theoretical_volatility(horizon)
        
        # Z-score para nivel de confianza
        from scipy import stats
        z = stats.norm.ppf((1 + confidence) / 2)
        
        # Intervalo
        lower = prediction - z * vol
        upper = prediction + z * vol
        
        return (lower, upper)
    
    def analyze_scaling(self) -> pd.DataFrame:
        """
        Analizar escalamiento de volatilidad observado vs teórico
        
        Returns:
            DataFrame con análisis comparativo
        """
        if not self.volatility_by_horizon:
            raise ValueError("No volatility data. Call fit() first.")
        
        rows = []
        for horizon in sorted(self.volatility_by_horizon.keys()):
            vol_obs = self.volatility_by_horizon[horizon]
            vol_theo = self.get_theoretical_volatility(horizon)
            ratio_obs = vol_obs / self.base_volatility
            ratio_theo = self.get_scaling_factor(self.base_horizon, horizon)
            
            rows.append({
                'horizon_days': horizon,
                'volatility_observed': vol_obs,
                'volatility_theoretical': vol_theo,
                'scaling_factor_observed': ratio_obs,
                'scaling_factor_theoretical': ratio_theo,
                'difference_pct': ((vol_obs - vol_theo) / vol_theo) * 100
            })
        
        return pd.DataFrame(rows)
    
    def correct_model_predictions(
        self,
        predictions: Dict[str, Dict],
        verbose: bool = True
    ) -> Dict[str, Dict]:
        """
        Corregir predicciones de modelos aplicando escalamiento de volatilidad
        
        Args:
            predictions: Diccionario con predicciones originales
                        {model_name: {'prediction': float, 'horizon_days': int}}
            verbose: Si True, muestra información de corrección
            
        Returns:
            Diccionario con predicciones corregidas
        """
        corrected = {}
        
        if verbose:
            print("="*80)
            print("CORRECCIÓN DE PREDICCIONES CON ESCALAMIENTO DE VOLATILIDAD")
            print("="*80)
            print()
        
        for model_name, pred_data in predictions.items():
            original_pred = pred_data['prediction']
            horizon = pred_data['horizon_days']
            
            # Obtener volatilidad observada y teórica
            vol_obs = self.get_observed_volatility(horizon)
            vol_theo = self.get_theoretical_volatility(horizon)
            
            # Calcular factor de corrección
            # Si la volatilidad observada es mayor, amplificar predicción
            if vol_obs is not None:
                correction_factor = vol_obs / vol_theo
            else:
                correction_factor = 1.0
            
            # Aplicar corrección
            corrected_pred = original_pred * correction_factor
            
            # Calcular intervalo de confianza
            lower, upper = self.get_confidence_interval(corrected_pred, horizon)
            
            corrected[model_name] = {
                **pred_data,
                'prediction_original': original_pred,
                'prediction_corrected': corrected_pred,
                'correction_factor': correction_factor,
                'confidence_interval_95': (lower, upper),
                'volatility_observed': vol_obs,
                'volatility_theoretical': vol_theo
            }
            
            if verbose:
                print(f"{model_name} ({horizon} días):")
                print(f"   Predicción original:  {original_pred:+.4f} ({original_pred*100:+.2f}%)")
                print(f"   Factor de corrección: {correction_factor:.3f}x")
                print(f"   Predicción corregida: {corrected_pred:+.4f} ({corrected_pred*100:+.2f}%)")
                print(f"   IC 95%: [{lower:+.4f}, {upper:+.4f}] ({lower*100:+.2f}%, {upper*100:+.2f}%)")
                print()
        
        return corrected


def create_scaler_from_dataset(dataset_path: str) -> VolatilityScaler:
    """
    Crear y ajustar escalador desde un archivo CSV
    
    Args:
        dataset_path: Ruta al dataset con targets
        
    Returns:
        VolatilityScaler ajustado
    """
    df = pd.read_csv(dataset_path)
    scaler = VolatilityScaler(base_horizon=1)
    scaler.fit(df)
    return scaler


if __name__ == "__main__":
    # Ejemplo de uso
    print("Creando escalador de volatilidad...")
    scaler = create_scaler_from_dataset('data/processed/fusion/dataset_stock.csv')
    
    print("\nAnálisis de escalamiento:")
    analysis = scaler.analyze_scaling()
    print(analysis.to_string(index=False))
    
    print("\n" + "="*80)
    print("Ejemplo: Corregir predicciones")
    print("="*80)
    
    # Predicciones simuladas (valores bajos típicos de modelos conservadores)
    predictions = {
        'sentiment_short_1d': {
            'prediction': -0.0224,
            'horizon_days': 1
        },
        'stock_medium_5d': {
            'prediction': 0.0223,
            'horizon_days': 5
        },
        'stock_long_21d': {
            'prediction': 0.0118,  # Muy conservador
            'horizon_days': 21
        }
    }
    
    corrected = scaler.correct_model_predictions(predictions)
