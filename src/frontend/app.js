document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Gather data from form
        const payload = {
            "Age": parseInt(document.getElementById('age').value),
            "Gender": document.getElementById('gender').value,
            "Customer Type": document.getElementById('customer_type').value,
            "Type of Travel": document.getElementById('travel_type').value,
            "Class": document.getElementById('class').value,
            "Flight Distance": parseInt(document.getElementById('distance').value),
            "Seat comfort": parseInt(document.getElementById('seat').value),
            "Departure/Arrival time convenient": 3, // default or add input
            "Food and drink": parseInt(document.getElementById('catering').value),
            "Gate location": 3,
            "Inflight wifi service": parseInt(document.getElementById('wifi').value),
            "Inflight entertainment": parseInt(document.getElementById('entertainment').value),
            "Online support": 3,
            "Ease of Online booking": 3,
            "On-board service": 3,
            "Leg room service": 3,
            "Baggage handling": 3,
            "Checkin service": parseInt(document.getElementById('checkin').value),
            "Cleanliness": 3,
            "Online boarding": 3,
            "Arrival Delay in Minutes": parseInt(document.getElementById('delay').value)
        };
        
        try {
            // First call Predict endpoint
            const res = await fetch('http://localhost:8000/predict/phase2', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            if (!res.ok) throw new Error('Error in prediction API');
            const data = await res.json();
            
            updateProbabilityGauge(data.probability, data.prediction);
            
            // Then call SHAP Explain endpoint
            const expRes = await fetch('http://localhost:8000/explain', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            if (expRes.ok) {
                const expData = await expRes.json();
                updateShapChart(expData.explanation);
            }
            
            updateRecommendation(data.probability);
            
        } catch (err) {
            console.error(err);
            alert("Error conectando con la API. Asegurate de que el backend esté corriendo.");
        }
    });
});

function updateProbabilityGauge(prob, predClass) {
    const probText = document.getElementById('prob-text');
    const badge = document.getElementById('risk-badge');
    const donut = document.getElementById('donut-progress');
    
    // Prob is satisfied (1)
    const probPct = Math.round(prob * 100);
    probText.innerText = probPct + "%";
    
    // circumference is 251.2
    const offset = (probPct / 100) * 251.2;
    donut.style.strokeDasharray = `${offset} 251.2`;
    
    if (probPct > 60) {
        badge.innerText = "Alta Satisfacción";
        badge.className = "text-sm px-2 py-1 bg-green-900/50 text-green-400 rounded border border-green-500/50 mt-2";
        donut.style.stroke = "#4ade80"; // green
    } else if (probPct > 40) {
        badge.innerText = "Satisfacción Media";
        badge.className = "text-sm px-2 py-1 bg-yellow-900/50 text-yellow-400 rounded border border-yellow-500/50 mt-2";
        donut.style.stroke = "#facc15"; // yellow
    } else {
        badge.innerText = "Alto Riesgo Insatisfacción";
        badge.className = "text-sm px-2 py-1 bg-red-900/50 text-red-400 rounded border border-red-500/50 mt-2";
        donut.style.stroke = "#f87171"; // red
    }
}

function updateShapChart(explanation) {
    const container = document.getElementById('shap-container');
    container.innerHTML = ''; // clear
    
    // take top 3 absolute impacts
    const top3 = explanation.slice(0, 3);
    const maxVal = Math.max(...top3.map(e => Math.abs(e.value)));
    
    top3.forEach(item => {
        const isPositive = item.value > 0;
        const widthPct = Math.max((Math.abs(item.value) / maxVal) * 100, 5); // min 5%
        
        // friendly name map
        let name = item.feature;
        if (name.includes('Inflight wifi')) name = "WiFi";
        if (name.includes('Arrival Delay')) name = "Retraso Llegada";
        if (name.includes('Class')) name = "Clase";
        if (name.includes('Customer Type')) name = "Tipo Cliente";
        
        container.innerHTML += `
            <div class="flex items-center gap-3">
                <span class="text-sm text-on-surface-variant w-24 text-right truncate" title="${item.feature}">${name}</span>
                <div class="flex-1 h-2 bg-surface-container-highest rounded-full overflow-hidden flex ${isPositive ? 'justify-start' : 'justify-end'}">
                    <div class="h-full ${isPositive ? 'bg-green-500' : 'bg-red-500'}" style="width: ${widthPct}%"></div>
                </div>
            </div>
        `;
    });
}

function updateRecommendation(prob) {
    const title = document.getElementById('action-title');
    const desc = document.getElementById('action-desc');
    
    if (prob > 0.6) {
        title.innerText = "Mantener Servicio Estándar";
        title.className = "font-bold text-green-400 uppercase";
        desc.innerText = "El perfil actual indica una alta probabilidad de satisfacción. No se requieren acciones compensatorias.";
    } else if (prob > 0.4) {
        title.innerText = "Vigilancia Activa";
        title.className = "font-bold text-yellow-400 uppercase";
        desc.innerText = "El pasajero está en el límite. Asegúrese de que el catering y la atención a bordo sean impecables para inclinar la balanza a favor.";
    } else {
        title.innerText = "Acción de Retención Inmediata";
        title.className = "font-bold text-red-400 uppercase";
        desc.innerText = "Alto riesgo de insatisfacción detectado. Se recomienda ofrecer voucher de descuento para futuro vuelo o upgrade gratuito si hay disponibilidad.";
    }
}
