import numpy as np
import random
from collections import defaultdict
from environment import TocaCorreContinuousEnv

# ===============================================================
#   DISCRETIZAR EL ESTADO CONTINUO - 🔥 MEJORADO
# ===============================================================

def discretizar_estado(state):
    """
    Convierte el estado continuo en estado discreto.
    
    🔥 MEJORADO: 
    - Discretización más fina (cada 1.5 unidades en lugar de 2)
    - Incluye distancia y ángulo discretizados
    """
    x_per, y_per, x_eva, y_eva, distancia, angulo, salto, ralent, trampa = state
    
    # 🔥 MEJORADO: discretizar posiciones en más zonas (cada 1.5 unidades)
    x_per_d = int(x_per / 1.5)
    y_per_d = int(y_per / 1.5)
    x_eva_d = int(x_eva / 1.5)
    y_eva_d = int(y_eva / 1.5)
    
    # 🔥 NUEVO: discretizar distancia en rangos
    if distancia < 2.0:
        dist_d = 0  # muy cerca
    elif distancia < 5.0:
        dist_d = 1  # cerca
    elif distancia < 10.0:
        dist_d = 2  # medio
    else:
        dist_d = 3  # lejos
    
    # 🔥 NUEVO: discretizar ángulo en 8 direcciones
    # angulo está en radianes [-π, π]
    angulo_grados = np.degrees(angulo) % 360
    angulo_d = int(angulo_grados / 45)  # 0-7

    return (x_per_d, y_per_d, x_eva_d, y_eva_d, dist_d, angulo_d, 
            int(salto), int(ralent), int(trampa))


# ===============================================================
#   POLÍTICA EPSILON-GREEDY - 🔥 MEJORADO
# ===============================================================

def seleccionar_accion(Q, estado, epsilon, n_acciones=9):
    """
    Selecciona acción usando epsilon-greedy.
    
    🔥 MEJORADO: Exploración más inteligente
    """
    if random.random() < epsilon:
        # Exploración aleatoria
        return random.randint(0, n_acciones - 1)
    else:
        # Explotación: mejor acción conocida
        valores = Q[estado]
        
        # 🔥 MEJORADO: Si hay empate, elegir aleatoriamente entre las mejores
        max_valor = np.max(valores)
        mejores_acciones = np.where(valores == max_valor)[0]
        
        if len(mejores_acciones) > 1:
            return np.random.choice(mejores_acciones)
        else:
            return mejores_acciones[0]


# ===============================================================
#   ENTRENAMIENTO Q-LEARNING - 🔥 PERFECCIONADO
# ===============================================================

def entrenar(q_episodes=5000, gamma=0.95, alpha=0.1, verbose=True):
    """
    Entrena ambos agentes usando Q-Learning.
    
    🔥 PERFECCIONADO:
    - Mejor decay de epsilon
    - Estadísticas completas
    - Alpha adaptativo
    - Early stopping si converge
    
    Returns:
        Q_per, Q_eva: Tablas Q entrenadas
        stats: Diccionario con estadísticas de entrenamiento
    """
    env = TocaCorreContinuousEnv()

    # Tablas Q de perseguidor y evasor
    Q_per = defaultdict(lambda: np.zeros(9))
    Q_eva = defaultdict(lambda: np.zeros(9))

    # Hiperparámetros
    epsilon = 1.0  # mucha exploración al inicio
    epsilon_min = 0.01  # 🔥 MEJORADO: epsilon mínimo más bajo
    epsilon_decay = 0.9995  # 🔥 MEJORADO: decay más suave
    
    alpha_inicial = alpha
    alpha_min = 0.01  # 🔥 NUEVO: learning rate mínimo
    alpha_decay = 0.9998  # 🔥 NUEVO: decay del learning rate

    # 🔥 NUEVO: Estadísticas de entrenamiento
    stats = {
        'recompensas_per': [],
        'recompensas_eva': [],
        'pasos_por_episodio': [],
        'intercambios_por_episodio': [],
        'capturas_por_episodio': [],
        'epsilon_history': [],
        'alpha_history': []
    }
    
    # 🔥 NUEVO: Ventanas móviles para convergencia
    ventana_recompensas_per = []
    ventana_recompensas_eva = []
    ventana_size = 100

    if verbose:
        print("=" * 60)
        print("  INICIANDO ENTRENAMIENTO Q-LEARNING")
        print("=" * 60)
        print(f"Episodios: {q_episodes}")
        print(f"Gamma (descuento): {gamma}")
        print(f"Alpha inicial: {alpha}")
        print(f"Epsilon inicial: {epsilon}")
        print("=" * 60)

    for episodio in range(q_episodes):

        estado = env.reset()
        estado_d = discretizar_estado(estado)

        total_r_per = 0
        total_r_eva = 0

        done = False

        for step in range(500):  # paso máximo por episodio

            # 1. Seleccionar acciones para ambos agentes
            a_per = seleccionar_accion(Q_per, estado_d, epsilon)
            a_eva = seleccionar_accion(Q_eva, estado_d, epsilon)

            # 2. Ejecutar step del entorno
            estado_nuevo, r_per, r_eva, done = env.step(a_per, a_eva)

            estado_nuevo_d = discretizar_estado(estado_nuevo)

            # 3. Actualizar Q para perseguidor
            Q_per[estado_d][a_per] += alpha * (
                r_per + gamma * np.max(Q_per[estado_nuevo_d]) - Q_per[estado_d][a_per]
            )
            
            # 4. Actualizar Q para evasor
            Q_eva[estado_d][a_eva] += alpha * (
                r_eva + gamma * np.max(Q_eva[estado_nuevo_d]) - Q_eva[estado_d][a_eva]
            )

            estado_d = estado_nuevo_d

            total_r_per += r_per
            total_r_eva += r_eva

            if done:
                break

        # 🔥 NUEVO: Guardar estadísticas
        episode_stats = env.get_stats()
        stats['recompensas_per'].append(total_r_per)
        stats['recompensas_eva'].append(total_r_eva)
        stats['pasos_por_episodio'].append(episode_stats['steps'])
        stats['intercambios_por_episodio'].append(episode_stats['intercambios'])
        stats['capturas_por_episodio'].append(episode_stats['capturas_perseguidor'])
        stats['epsilon_history'].append(epsilon)
        stats['alpha_history'].append(alpha)

        # 🔥 NUEVO: Ventana móvil para convergencia
        ventana_recompensas_per.append(total_r_per)
        ventana_recompensas_eva.append(total_r_eva)
        
        if len(ventana_recompensas_per) > ventana_size:
            ventana_recompensas_per.pop(0)
            ventana_recompensas_eva.pop(0)

        # Disminuir epsilon y alpha
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        alpha = max(alpha * alpha_decay, alpha_min)

        # Mostrar progreso
        if verbose and episodio % 100 == 0:
            promedio_per = np.mean(ventana_recompensas_per) if ventana_recompensas_per else 0
            promedio_eva = np.mean(ventana_recompensas_eva) if ventana_recompensas_eva else 0
            
            print(f"Ep {episodio:5d} | "
                  f"R_per: {total_r_per:7.1f} (avg: {promedio_per:7.1f}) | "
                  f"R_eva: {total_r_eva:7.1f} (avg: {promedio_eva:7.1f}) | "
                  f"Steps: {episode_stats['steps']:3d} | "
                  f"Intercambios: {episode_stats['intercambios']} | "
                  f"ε: {epsilon:.4f} | "
                  f"α: {alpha:.4f}")

        # 🔥 NUEVO: Early stopping si converge
        if episodio > 1000 and len(ventana_recompensas_per) == ventana_size:
            varianza_per = np.var(ventana_recompensas_per)
            varianza_eva = np.var(ventana_recompensas_eva)
            
            # Si la varianza es muy baja, el aprendizaje ha convergido
            if varianza_per < 1000 and varianza_eva < 1000:
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"  CONVERGENCIA DETECTADA EN EPISODIO {episodio}")
                    print(f"{'='*60}")
                break

    if verbose:
        print("\n" + "=" * 60)
        print("  ENTRENAMIENTO TERMINADO")
        print("=" * 60)
        print(f"Episodios completados: {episodio + 1}")
        print(f"Estados explorados (Perseguidor): {len(Q_per)}")
        print(f"Estados explorados (Evasor): {len(Q_eva)}")
        print(f"Epsilon final: {epsilon:.4f}")
        print(f"Alpha final: {alpha:.4f}")
        
        # 🔥 NUEVO: Estadísticas finales
        if len(stats['recompensas_per']) > 0:
            print(f"\nÚltimos 100 episodios:")
            ultimos_100_per = stats['recompensas_per'][-100:]
            ultimos_100_eva = stats['recompensas_eva'][-100:]
            print(f"  Recompensa promedio Perseguidor: {np.mean(ultimos_100_per):.1f}")
            print(f"  Recompensa promedio Evasor: {np.mean(ultimos_100_eva):.1f}")
            print(f"  Pasos promedio: {np.mean(stats['pasos_por_episodio'][-100:]):.1f}")
            print(f"  Intercambios promedio: {np.mean(stats['intercambios_por_episodio'][-100:]):.2f}")
        
        print("=" * 60 + "\n")

    return Q_per, Q_eva, stats


# ===============================================================
#   🔥 NUEVO: GUARDAR ESTADÍSTICAS
# ===============================================================

def guardar_estadisticas(stats, filename='training_stats.npy'):
    """Guarda las estadísticas de entrenamiento"""
    np.save(filename, stats)
    print(f"Estadísticas guardadas en {filename}")


# ===============================================================
#   🔥 NUEVO: CARGAR ESTADÍSTICAS
# ===============================================================

def cargar_estadisticas(filename='training_stats.npy'):
    """Carga las estadísticas de entrenamiento"""
    try:
        stats = np.load(filename, allow_pickle=True).item()
        print(f"Estadísticas cargadas desde {filename}")
        return stats
    except:
        print(f"No se pudo cargar {filename}")
        return None