import numpy as np
import random
from collections import defaultdict
from environment import TocaCorreContinuousEnv3Agentes

# ===============================================================
#   DISCRETIZAR EL ESTADO CONTINUO
# ===============================================================

def discretizar_estado(estados_continuos):
    """
    Convierte estados continuos (lista de 3) en discretos.
    estados_continuos: lista de 3 arrays (uno por agente)
    Retorna: lista de 3 tuplas discretas
    """
    estados_discretos = []
    
    for state in estados_continuos:
        x_agente, y_agente, x_presa, y_presa, dist_presa, ang_presa, trampa, dist_pers = state
        
        # Discretizar posiciones
        x_d = int(x_agente / 1.5)
        y_d = int(y_agente / 1.5)
        x_presa_d = int(x_presa / 1.5)
        y_presa_d = int(y_presa / 1.5)
        
        # Discretizar distancia a presa
        if dist_presa < 2.0:
            dist_presa_d = 0
        elif dist_presa < 5.0:
            dist_presa_d = 1
        elif dist_presa < 10.0:
            dist_presa_d = 2
        else:
            dist_presa_d = 3
        
        # Discretizar ángulo a presa
        angulo_grados = np.degrees(ang_presa) % 360
        angulo_d = int(angulo_grados / 45)
        
        # Discretizar distancia al perseguidor
        if dist_pers < 2.0:
            dist_pers_d = 0
        elif dist_pers < 5.0:
            dist_pers_d = 1
        else:
            dist_pers_d = 2
        
        estado_d = (x_d, y_d, x_presa_d, y_presa_d, dist_presa_d, 
                   angulo_d, int(trampa), dist_pers_d)
        
        estados_discretos.append(estado_d)
    
    return estados_discretos


# ===============================================================
#   POLÍTICA EPSILON-GREEDY
# ===============================================================

def seleccionar_accion(Q, estado, epsilon, n_acciones=9):
    """
    Selecciona acción usando epsilon-greedy.
    """
    if random.random() < epsilon:
        return random.randint(0, n_acciones - 1)
    else:
        valores = Q[estado]
        max_valor = np.max(valores)
        mejores_acciones = np.where(valores == max_valor)[0]
        
        if len(mejores_acciones) > 1:
            return np.random.choice(mejores_acciones)
        else:
            return mejores_acciones[0]


# ===============================================================
#   ENTRENAMIENTO Q-LEARNING - 3 AGENTES
# ===============================================================

def entrenar(q_episodes=5000, gamma=0.95, alpha=0.1, verbose=True):
    """
    Entrena 3 agentes en ciclo usando Q-Learning.
    """
    env = TocaCorreContinuousEnv3Agentes()

    # Tablas Q para cada agente
    Q_agentes = [
        defaultdict(lambda: np.zeros(9)),  # Agente 0
        defaultdict(lambda: np.zeros(9)),  # Agente 1
        defaultdict(lambda: np.zeros(9))   # Agente 2
    ]

    # Hiperparámetros
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.9995
    
    alpha_inicial = alpha
    alpha_min = 0.01
    alpha_decay = 0.9998

    # Estadísticas
    stats = {
        'recompensas_agente': [[], [], []],
        'pasos_por_episodio': [],
        'capturas_totales': [],
        'epsilon_history': [],
        'alpha_history': []
    }
    
    ventana_recompensas = [[], [], []]
    ventana_size = 100

    if verbose:
        print("=" * 70)
        print("  ENTRENAMIENTO Q-LEARNING - 3 AGENTES EN CICLO")
        print("=" * 70)
        print(f"Configuración:")
        print(f"  Episodios: {q_episodes}")
        print(f"  Gamma: {gamma}")
        print(f"  Alpha: {alpha}")
        print(f"  Epsilon: {epsilon}")
        print("  Estructura: A->B, B->C, C->A")
        print("=" * 70)

    for episodio in range(q_episodes):

        estados = env.reset()
        estados_d = discretizar_estado(estados)

        total_r_agentes = [0.0, 0.0, 0.0]
        done = False

        for step in range(500):

            # 1. Seleccionar acciones
            acciones = []
            for i in range(3):
                accion = seleccionar_accion(Q_agentes[i], estados_d[i], epsilon)
                acciones.append(accion)

            # 2. Ejecutar step
            estados_nuevos, recompensas, done = env.step(acciones)
            estados_nuevos_d = discretizar_estado(estados_nuevos)

            # 3. Actualizar Q para cada agente
            for i in range(3):
                Q_agentes[i][estados_d[i]][acciones[i]] += alpha * (
                    recompensas[i] + gamma * np.max(Q_agentes[i][estados_nuevos_d[i]]) 
                    - Q_agentes[i][estados_d[i]][acciones[i]]
                )
                
                total_r_agentes[i] += recompensas[i]

            estados_d = estados_nuevos_d

            if done:
                break

        # Guardar estadísticas
        episode_stats = env.get_stats()
        for i in range(3):
            stats['recompensas_agente'][i].append(total_r_agentes[i])
            ventana_recompensas[i].append(total_r_agentes[i])
            if len(ventana_recompensas[i]) > ventana_size:
                ventana_recompensas[i].pop(0)
        
        stats['pasos_por_episodio'].append(episode_stats['steps'])
        stats['capturas_totales'].append(episode_stats['capturas_total'])
        stats['epsilon_history'].append(epsilon)
        stats['alpha_history'].append(alpha)

        # Disminuir hiperparámetros
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        alpha = max(alpha * alpha_decay, alpha_min)

        # Mostrar progreso
        if verbose and episodio % 100 == 0:
            prom_r = [np.mean(ventana_recompensas[i]) if ventana_recompensas[i] else 0 
                     for i in range(3)]
            
            print(f"Ep {episodio:5d} | "
                  f"R0: {total_r_agentes[0]:7.1f} (avg: {prom_r[0]:7.1f}) | "
                  f"R1: {total_r_agentes[1]:7.1f} (avg: {prom_r[1]:7.1f}) | "
                  f"R2: {total_r_agentes[2]:7.1f} (avg: {prom_r[2]:7.1f}) | "
                  f"Steps: {episode_stats['steps']:3d} | "
                  f"Capturas: {episode_stats['capturas_total']} | "
                  f"ε: {epsilon:.4f}")

        # Early stopping
        if episodio > 1000 and all(len(v) == ventana_size for v in ventana_recompensas):
            varianzas = [np.var(v) for v in ventana_recompensas]
            if all(var < 1000 for var in varianzas):
                if verbose:
                    print(f"\n{'='*70}")
                    print(f"  CONVERGENCIA DETECTADA EN EPISODIO {episodio}")
                    print(f"{'='*70}")
                break

    if verbose:
        print("\n" + "=" * 70)
        print("  ENTRENAMIENTO TERMINADO")
        print("=" * 70)
        print(f"Episodios completados: {episodio + 1}")
        for i in range(3):
            print(f"Estados explorados (Agente {i}): {len(Q_agentes[i])}")
        print(f"Epsilon final: {epsilon:.4f}")
        print(f"Alpha final: {alpha:.4f}")
        
        if len(stats['recompensas_agente'][0]) > 0:
            print(f"\nÚltimos 100 episodios:")
            for i in range(3):
                ultimos = stats['recompensas_agente'][i][-100:]
                print(f"  Agente {i}: {np.mean(ultimos):.1f} (prom)")
            print(f"  Pasos promedio: {np.mean(stats['pasos_por_episodio'][-100:]):.1f}")
            print(f"  Capturas totales: {np.sum(stats['capturas_totales'][-100:])}")
        
        print("=" * 70 + "\n")

    return Q_agentes, stats


# ===============================================================
#   GUARDAR/CARGAR ESTADÍSTICAS
# ===============================================================

def guardar_estadisticas(stats, filename='training_stats_3agentes.npy'):
    """Guarda las estadísticas de entrenamiento"""
    np.save(filename, stats)
    print(f"Estadísticas guardadas en {filename}")


def cargar_estadisticas(filename='training_stats_3agentes.npy'):
    """Carga las estadísticas de entrenamiento"""
    try:
        stats = np.load(filename, allow_pickle=True).item()
        print(f"Estadísticas cargadas desde {filename}")
        return stats
    except:
        print(f"No se pudo cargar {filename}")
        return None