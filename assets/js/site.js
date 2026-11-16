/* Small enhancements; the site's text, links and figures also work without JS. */
(() => {
  'use strict';
  document.documentElement.classList.add('js');

  const header = document.querySelector('.site-header');
  const navToggle = document.querySelector('[data-nav-toggle]');
  const navigation = document.getElementById('site-navigation');
  if (header && navToggle && navigation) {
    const closeNavigation = () => {
      header.dataset.menuOpen = 'false';
      navToggle.setAttribute('aria-expanded', 'false');
    };
    navToggle.addEventListener('click', () => {
      const open = navToggle.getAttribute('aria-expanded') !== 'true';
      header.dataset.menuOpen = String(open);
      navToggle.setAttribute('aria-expanded', String(open));
    });
    navigation.addEventListener('click', event => {
      if (event.target.closest('a')) closeNavigation();
    });
    document.addEventListener('keydown', event => {
      if (event.key === 'Escape' && header.dataset.menuOpen === 'true') {
        closeNavigation();
        navToggle.focus();
      }
    });
    document.addEventListener('click', event => {
      if (!header.contains(event.target)) closeNavigation();
    });
  }

  const feedback = document.getElementById('copy-feedback');
  async function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      try { await navigator.clipboard.writeText(text); return; } catch (_) { /* Use the local-file fallback. */ }
    }
    const active = document.activeElement;
    const area = document.createElement('textarea');
    area.value = text;
    area.setAttribute('readonly', '');
    area.style.cssText = 'position:fixed;left:-9999px;top:0';
    document.body.append(area);
    area.select();
    let copied = false;
    try { copied = document.execCommand('copy'); } finally { area.remove(); }
    if (active && active.focus) active.focus();
    if (!copied) throw new Error('Clipboard access unavailable');
  }
  document.querySelectorAll('[data-copy]').forEach(button => {
    button.addEventListener('click', async () => {
      const code = button.closest('.code-block')?.querySelector('pre code, pre');
      if (!code) return;
      const originalLabel = button.textContent;
      button.disabled = true;
      try {
        await copyText(code.textContent.trim());
        button.textContent = 'Copied';
        if (feedback) feedback.textContent = 'Copied to clipboard.';
      } catch (_) {
        button.textContent = 'Select text';
        const range = document.createRange();
        range.selectNodeContents(code);
        const selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
        if (feedback) feedback.textContent = 'Clipboard unavailable. The text is selected; copy it with your keyboard.';
      }
      window.setTimeout(() => { button.textContent = originalLabel; button.disabled = false; }, 2000);
    });
  });

  const dialog = document.getElementById('image-dialog');
  const dialogImage = document.getElementById('dialog-image');
  const dialogTitle = document.getElementById('dialog-title');
  if (dialog && typeof dialog.showModal === 'function' && dialogImage && dialogTitle) {
    document.querySelectorAll('a[data-lightbox]').forEach(link => {
      link.addEventListener('click', event => {
        if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
        event.preventDefault();
        const image = link.querySelector('img');
        dialogImage.src = link.href;
        dialogImage.alt = image?.alt || link.dataset.lightbox || 'Scientific figure';
        dialogTitle.textContent = link.dataset.lightbox || image?.alt || 'Scientific figure';
        dialog.showModal();
      });
    });
    dialog.querySelector('[data-close-dialog]')?.addEventListener('click', () => dialog.close());
    dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
    dialog.addEventListener('close', () => { dialogImage.removeAttribute('src'); });
  }

  const catalog = document.querySelector('[data-case-catalog]');
  if (!catalog) return;
  const cards = Array.from(document.querySelectorAll('[data-case-card]'));
  const panels = Array.from(document.querySelectorAll('[data-case-detail]'));
  const search = document.getElementById('case-search');
  const sensorFilter = document.getElementById('sensor-filter');
  const count = document.getElementById('case-count');
  const empty = document.getElementById('catalog-empty');
  const filterForm = document.querySelector('[data-case-filters]');
  const selectedIds = new Set(panels.map(panel => panel.id));
  if (!cards.length || !panels.length) return;
  if (filterForm) { filterForm.hidden = false; filterForm.addEventListener('submit', event => event.preventDefault()); }

  function selectCase(id, { updateAddress = true, moveFocus = false } = {}) {
    if (!selectedIds.has(id)) return;
    panels.forEach(panel => { panel.hidden = panel.id !== id; });
    cards.forEach(card => {
      if (card.dataset.caseCard === id) card.setAttribute('aria-current', 'true');
      else card.removeAttribute('aria-current');
    });
    if (updateAddress) {
      try {
        const url = new URL(window.location.href);
        url.searchParams.set('case', id);
        url.hash = '';
        window.history.replaceState({ case: id }, '', url);
      } catch (_) { /* Local-file browsing still supports selection. */ }
    }
    if (moveFocus) {
      const heading = document.getElementById(id).querySelector('h2');
      heading?.focus({ preventScroll: true });
      document.getElementById(id).scrollIntoView({ behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth', block: 'start' });
    }
  }
  function caseFromAddress() {
    let hash = window.location.hash.slice(1);
    try { hash = decodeURIComponent(hash); } catch (_) { /* Ignore malformed URL escapes. */ }
    if (selectedIds.has(hash)) return hash;
    const destination = document.getElementById(hash);
    if (destination?.closest('[data-case-detail]')) return destination.closest('[data-case-detail]').id;
    const requested = new URLSearchParams(window.location.search).get('case');
    if (selectedIds.has(requested)) return requested;
    return panels[0].id;
  }
  cards.forEach(card => {
    card.addEventListener('click', event => {
      if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      selectCase(card.dataset.caseCard, { moveFocus: true });
    });
  });
  selectCase(caseFromAddress(), { updateAddress: false });
  window.addEventListener('popstate', () => selectCase(caseFromAddress(), { updateAddress: false }));
  window.addEventListener('hashchange', () => selectCase(caseFromAddress(), { updateAddress: false }));

  function applyFilters() {
    const term = (search?.value || '').trim().toLocaleLowerCase();
    const sensor = sensorFilter?.value || '';
    const visible = [];
    cards.forEach(card => {
      const matchesText = !term || card.dataset.search.toLocaleLowerCase().includes(term);
      const matchesSensor = !sensor || card.dataset.sensors.split('|').includes(sensor);
      card.hidden = !(matchesText && matchesSensor);
      if (!card.hidden) visible.push(card);
    });
    if (count) count.textContent = `${visible.length} of ${cards.length} case studies`;
    if (empty) empty.hidden = visible.length !== 0;
    const active = visible.find(card => card.getAttribute('aria-current') === 'true');
    if (visible.length) selectCase((active || visible[0]).dataset.caseCard);
    panels.forEach(panel => {
      const matchingCard = visible.find(card => card.dataset.caseCard === panel.id);
      if (!matchingCard) panel.hidden = true;
    });
  }
  search?.addEventListener('input', applyFilters);
  sensorFilter?.addEventListener('change', applyFilters);
  document.querySelector('[data-reset-filters]')?.addEventListener('click', () => {
    if (search) search.value = '';
    if (sensorFilter) sensorFilter.value = '';
    applyFilters();
  });

  document.querySelectorAll('[data-figure-tabs]').forEach(group => {
    const tabs = Array.from(group.querySelectorAll('[data-figure-tab]'));
    if (!tabs.length) return;
    group.setAttribute('role', 'tablist');
    function selectFigure(tab, focus = false) {
      tabs.forEach(item => {
        const selected = item === tab;
        item.setAttribute('role', 'tab');
        item.setAttribute('aria-selected', String(selected));
        item.tabIndex = selected ? 0 : -1;
        const panel = document.getElementById(item.getAttribute('aria-controls'));
        if (panel) { panel.hidden = !selected; panel.setAttribute('role', 'tabpanel'); panel.setAttribute('aria-labelledby', item.id); }
      });
      if (focus) tab.focus();
    }
    tabs.forEach((tab, index) => {
      tab.addEventListener('click', () => selectFigure(tab));
      tab.addEventListener('keydown', event => {
        let target;
        if (event.key === 'ArrowRight') target = (index + 1) % tabs.length;
        else if (event.key === 'ArrowLeft') target = (index - 1 + tabs.length) % tabs.length;
        else if (event.key === 'Home') target = 0;
        else if (event.key === 'End') target = tabs.length - 1;
        else return;
        event.preventDefault(); selectFigure(tabs[target], true);
      });
    });
    function selectFigureFromAddress() {
      let hash = window.location.hash.slice(1);
      try { hash = decodeURIComponent(hash); } catch (_) { return; }
      const destination = document.getElementById(hash);
      const panel = destination?.closest('[data-figure-panel]');
      const tab = panel && tabs.find(item => item.getAttribute('aria-controls') === panel.id);
      if (tab) selectFigure(tab);
    }
    selectFigure(tabs[0]);
    selectFigureFromAddress();
    window.addEventListener('hashchange', selectFigureFromAddress);
  });
})();
