import numpy as np

def calcular_recompensas_3agentes(env, prev_posiciones, toques, en_trampa):
    """
    Calcula recompensas para 3 agentes en ciclo (A->B, B->C, C->A)
    
    Recompensas basadas en:
    - Acercamiento a la presa
    - Distancia al perseguidor
    - Uso de trampas
    - Captura
    """
    recompensas = [0.0, 0.0, 0.0]
    
    for i in range(3):
        cazador = env.agentes[i]
        presa = env.agentes[(i + 1) % 3]
        perseguidor = env.agentes[(i - 1) % 3]
        
        # ======================================================
        #   1. RECOMPENSA BASADA EN DISTANCIA A PRESA
        # ======================================================
        dist_previa = np.linalg.norm(np.array([
            prev_posiciones[i][0] - prev_posiciones[(i + 1) % 3][0],
            prev_posiciones[i][1] - prev_posiciones[(i + 1) % 3][1]
        ]))
        
        dist_actual = np.linalg.norm(np.array([
            cazador.x - presa.x,
            cazador.y - presa.y
        ]))
        
        cambio_distancia = dist_previa - dist_actual
        
        if cambio_distancia > 0:  # Se acercó a la presa
            recompensas[i] += 10 * cambio_distancia
        else:  # Se alejó de la presa
            recompensas[i] -= 5 * abs(cambio_distancia)
        
        # Bonificación por distancia crítica
        if dist_actual < 1.5:
            recompensas[i] += 15
        elif dist_actual > 10.0:
            recompensas[i] -= 3
        
        # ======================================================
        #   2. RECOMPENSA POR ALEJAR DEL PERSEGUIDOR
        # ======================================================
        dist_pers_previa = np.linalg.norm(np.array([
            prev_posiciones[i][0] - prev_posiciones[(i - 1) % 3][0],
            prev_posiciones[i][1] - prev_posiciones[(i - 1) % 3][1]
        ]))
        
        dist_pers_actual = np.linalg.norm(np.array([
            cazador.x - perseguidor.x,
            cazador.y - perseguidor.y
        ]))
        
        cambio_dist_pers = dist_pers_actual - dist_pers_previa
        
        if cambio_dist_pers > 0:  # Se alejó del perseguidor
            recompensas[i] += 8 * cambio_dist_pers
        else:  # Se acercó al perseguidor
            recompensas[i] -= 6 * abs(cambio_dist_pers)
        
        # Penalización si está muy cerca del perseguidor
        if dist_pers_actual < 1.5:
            recompensas[i] -= 20
        elif dist_pers_actual > 10.0:
            recompensas[i] += 3
        
        # ======================================================
        #   3. TRAMPA
        # ======================================================
        if en_trampa[i]:
            recompensas[i] -= 8
        
        # Recompensa si pone trampa siendo perseguido
        if cazador.trampa_activa and dist_pers_actual < 5.0:
            recompensas[i] += 5
        
        # Penalización si pone trampa sin peligro
        if cazador.trampa_activa and dist_pers_actual > 8.0:
            recompensas[i] -= 2
        
        # ======================================================
        #   4. ZONAS ESPECIALES
        # ======================================================
        cx, cy = cazador.x, cazador.y
        
        # Zonas rápidas
        for zx, zy, boost in env.zonas_acelerar:
            if np.linalg.norm([cx - zx, cy - zy]) < 1.0:
                recompensas[i] += 3
        
        # Zonas lentas
        for zx, zy, slow in env.zonas_lentas:
            if np.linalg.norm([cx - zx, cy - zy]) < 1.0:
                recompensas[i] -= 3
        
        # ======================================================
        #   5. CAPTURA (TOCA)
        # ======================================================
        if toques[i]:
            recompensas[i] += 200  # Capturó a presa
        
        # Penalización si fue capturado
        if toques[(i - 1) % 3]:
            recompensas[i] -= 150
        
        # Bonus si aguantó muchos pasos
        if not toques[(i - 1) % 3] and env.steps > 200:
            recompensas[i] += 20
        
        # ======================================================
        #   6. PENALIZACIÓN POR INACTIVIDAD
        # ======================================================
        recompensas[i] -= 0.5

    return recompensas