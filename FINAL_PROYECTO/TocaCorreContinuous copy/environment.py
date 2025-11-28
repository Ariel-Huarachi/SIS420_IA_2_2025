import random
import numpy as np

from agents import Perseguidor, Evasor
from maze import Maze
from rewards import calcular_recompensas
from step_logic import (
    mover_agente,
    aplicar_zonas,
    detectar_toca,
    aplicar_trampa,
    intercambiar_roles_mejorado,
    detectar_atasco_global,
    desatascar_agentes
)

class TocaCorreContinuousEnv:

    def __init__(self):
        self.maze = Maze()

        # radio por defecto de los agentes
        self.agent_radius = 0.3

        # velocidades base
        self.speed_per = 1.0
        self.speed_eva = 1.0

        # zonas especiales
        self.zonas_acelerar = [(5, 5, 1.5), (12, 10, 1.5)]
        self.zonas_lentas = [(8, 3, 0.5), (15, 15, 0.6)]

        # agentes
        self.perseguidor = None
        self.evasor = None

        self.done = False
        self.steps = 0
        
        # Estadísticas del episodio
        self.num_intercambios = 0
        self.capturas_perseguidor = 0
        self.capturas_evasor = 0
        self.total_distancia_recorrida_per = 0
        self.total_distancia_recorrida_eva = 0

    # ==========================================================
    #   POSICIÓN ALEATORIA NO EN PAREDES
    # ==========================================================
    def _random_valid_position(self):
        max_intentos = 100
        for _ in range(max_intentos):
            x = random.uniform(1, self.maze.width - 1)
            y = random.uniform(1, self.maze.height - 1)
            if not self.maze.check_collision(x, y, self.agent_radius):
                return (x, y)
        return (self.maze.width / 2, self.maze.height / 2)

    # ==========================================================
    #   RESET DEL ENTORNO
    # ==========================================================
    def reset(self):
        x1, y1 = self._random_valid_position()
        x2, y2 = self._random_valid_position()

        intentos = 0
        while np.linalg.norm([x1 - x2, y1 - y2]) < 3.0 and intentos < 50:
            x2, y2 = self._random_valid_position()
            intentos += 1

        self.perseguidor = Perseguidor(x1, y1, radius=self.agent_radius, speed=self.speed_per)
        self.evasor = Evasor(x2, y2, radius=self.agent_radius, speed=self.speed_eva)

        self.perseguidor.reset_salto()
        self.evasor.trampa_activa = None
        self.evasor.trampas_puestas = 0
        self.perseguidor.saltos_usados = 0

        # resetear estadísticas
        self.steps = 0
        self.done = False
        self.num_intercambios = 0
        self.capturas_perseguidor = 0
        self.capturas_evasor = 0
        self.total_distancia_recorrida_per = 0
        self.total_distancia_recorrida_eva = 0

        return self._get_state()

    # ==========================================================
    #   OBTENER ESTADO (PARA RL)
    # ==========================================================
    def _get_state(self):
        dx = self.evasor.x - self.perseguidor.x
        dy = self.evasor.y - self.perseguidor.y
        distancia = np.linalg.norm([dx, dy])
        angulo = np.arctan2(dy, dx)
        
        return np.array([
            self.perseguidor.x,
            self.perseguidor.y,
            self.evasor.x,
            self.evasor.y,
            distancia,
            angulo,
            1 if self.perseguidor.salto_disponible else 0,
            1 if self._per_in_trampa() else 0,
            1 if self.evasor.trampa_activa is not None else 0
        ], dtype=float)

    def _per_in_trampa(self):
        if self.evasor.trampa_activa is None:
            return False

        tx, ty = self.evasor.trampa_activa
        px, py = self.perseguidor.get_position()

        return np.linalg.norm([px - tx, py - ty]) < 0.7

    # ==========================================================
    #   STEP – LÓGICA COMPLETA CON ANTI-ATASCOS
    # ==========================================================
    def step(self, accion_per, accion_eva):

        self.steps += 1
        done = False

        prev_pos_per = self.perseguidor.get_position()
        prev_pos_eva = self.evasor.get_position()

        # 1. Perseguidor
        if accion_per == 8:
            self.perseguidor.salto(self.evasor.get_position(), self.maze)
        else:
            mover_agente(self.perseguidor, accion_per, self.maze)

        nueva_pos_per = self.perseguidor.get_position()
        self.total_distancia_recorrida_per += np.linalg.norm(np.array(nueva_pos_per) - np.array(prev_pos_per))

        # 2. Evasor
        if accion_eva == 8:
            self.evasor.poner_trampa()
        else:
            mover_agente(self.evasor, accion_eva, self.maze)

        nueva_pos_eva = self.evasor.get_position()
        self.total_distancia_recorrida_eva += np.linalg.norm(np.array(nueva_pos_eva) - np.array(prev_pos_eva))

        # 3. Trampa
        en_trampa = aplicar_trampa(self.evasor, self.perseguidor, self)
        self.evasor.actualizar_trampa()

        # 4. Zonas especiales
        aplicar_zonas(self.perseguidor, self)
        aplicar_zonas(self.evasor, self)

        # 🔥 5. DETECCIÓN Y CORRECCIÓN DE ATASCOS (cada 20 pasos)
        if self.steps % 20 == 0:
            if detectar_atasco_global(self):
                desatascar_agentes(self)

        # 6. Detección de toca
        hubo_toca = detectar_toca(self.perseguidor, self.evasor)

        # 7. Recompensas
        r_per, r_eva = calcular_recompensas(
            prev_pos_per, prev_pos_eva,
            self.perseguidor, self.evasor,
            self,
            hubo_toca,
            en_trampa
        )

        # 8. Intercambio de roles
        if hubo_toca:
            self.num_intercambios += 1
            self.capturas_perseguidor += 1
            
            # 🔥 DEBUG: Mostrar captura
            dist_captura = np.linalg.norm([self.perseguidor.x - self.evasor.x,
                                          self.perseguidor.y - self.evasor.y])
        #    print(f"🎯 CAPTURA #{self.num_intercambios} en paso {self.steps}! Dist={dist_captura:.2f}")
            
            intercambiar_roles_mejorado(self)
            done = False

        # El episodio continúa indefinidamente
        done = False

        estado_nuevo = self._get_state()

        return estado_nuevo, r_per, r_eva, done

    # ==========================================================
    #   ESTADÍSTICAS
    # ==========================================================
    def get_stats(self):
        return {
            'steps': self.steps,
            'intercambios': self.num_intercambios,
            'capturas_perseguidor': self.capturas_perseguidor,
            'distancia_per': self.total_distancia_recorrida_per,
            'distancia_eva': self.total_distancia_recorrida_eva,
            'saltos_usados': self.perseguidor.saltos_usados if self.perseguidor else 0,
            'trampas_puestas': self.evasor.trampas_puestas if self.evasor else 0
        }